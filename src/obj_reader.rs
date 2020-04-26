use bmp::{BmpResult, Image};
use bytemuck::{Pod, Zeroable};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::str::SplitWhitespace;

pub struct IndexTuoel {
    vertex: [u16; 3],
    texture: Option<[u16; 3]>,
    normals: [u16; 3],
    style: Option<u8>,
}

pub struct ObjFile {
    pub vertices: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub textures: Vec<[f32; 2]>,
    pub indices: Vec<IndexTuoel>,
    pub materials: Option<Vec<Material>>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    texture: [f32; 2],
    style: u32,
}

unsafe impl Pod for Vertex {}
unsafe impl Zeroable for Vertex {}

pub type Mesh = Vec<Vertex>;

impl ObjFile {
    pub fn flatten(&self) -> Mesh {
        let mut mesh = Vec::with_capacity(self.indices.len() * 3);

        for IndexTuoel {
            vertex,
            normals,
            texture,
            style,
        } in self.indices.iter()
        {
            mesh.push(Vertex {
                position: self.vertices[vertex[0] as usize],
                normal: self.normals[normals[0] as usize],
                texture: texture
                    .map(|tex| self.textures[tex[0] as usize])
                    .unwrap_or_else(|| [0f32, 0f32]),
                style: style.unwrap_or_else(|| 255) as u32,
            });
            mesh.push(Vertex {
                position: self.vertices[vertex[1] as usize],
                normal: self.normals[normals[1] as usize],
                texture: texture
                    .map(|tex| self.textures[tex[1] as usize])
                    .unwrap_or_else(|| [0f32, 0f32]),
                style: style.unwrap_or_else(|| 255) as u32,
            });
            mesh.push(Vertex {
                position: self.vertices[vertex[2] as usize],
                normal: self.normals[normals[2] as usize],
                texture: texture
                    .map(|tex| self.textures[tex[2] as usize])
                    .unwrap_or_else(|| [0f32, 0f32]),
                style: style.unwrap_or_else(|| 255) as u32,
            });
        }

        mesh
    }

    fn read_mtl_lib(filename: &str) -> Result<Vec<Material>, ()> {
        let file = File::open(filename).unwrap(); //.map_err(|_| ())?;
        let mut reader = BufReader::new(file);

        let mut materials = Vec::new();

        let mut ambient_color: Option<[f32; 3]> = None;
        let mut diffuse_color: Option<[f32; 3]> = None;
        let mut specular_color: Option<[f32; 3]> = None;
        let mut specular_exponent: Option<u16> = None;
        let mut texture: Option<Image> = None;
        let mut name: Option<String> = None;

        loop {
            let mut line = String::new();
            let read = reader.read_line(&mut line).unwrap(); //.map_err(|_| ())?;

            if read == 0 || line.starts_with("newmtl ") {
                if let Some(name) = name.take() {
                    materials.push(Material {
                        name,
                        ambient_color: ambient_color.take(),
                        diffuse_color: diffuse_color.take(),
                        specular_color: specular_color.take(),
                        specular_exponent: specular_exponent.take(),
                        texture: texture.take(),
                    })
                }

                if read == 0 {
                    break;
                } else {
                    name = Some(line.as_str()[7..].trim().to_string());
                }
            }

            if line.starts_with("Ka ") {
                ambient_color = Some(Self::parse_vec3_line(&line).unwrap());
            } else if line.starts_with("Kd ") {
                diffuse_color = Some(Self::parse_vec3_line(&line).unwrap());
            } else if line.starts_with("Ks ") {
                specular_color = Some(Self::parse_vec3_line(&line).unwrap());
            } else if line.starts_with("Ns ") {
                specular_exponent = Some(line.as_str()[3..].trim().parse().unwrap());
            //.map_err(|_| ())?);
            } else if line.starts_with("map_Kd ") {
                texture = Some(bmp::open(&line[7..].trim()).unwrap());
            }
        }
        Ok(materials)
    }

    pub fn read(filename: &str) -> Result<ObjFile, ()> {
        let file = File::open(filename).map_err(|_| ())?;
        let mut reader = BufReader::new(file);

        let mut vertices = Vec::new();
        let mut normals = Vec::new();
        let mut textures = Vec::new();
        let mut indices = Vec::new();
        let mut indices_materials = Vec::new();

        let mut materials: Option<Vec<Material>> = None;
        let mut current_material_index: Option<u8> = None;

        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).map_err(|_| ())? == 0 {
                break;
            }
            if line.starts_with("v ") {
                vertices.push(ObjFile::parse_vec3_line(line.as_str())?)
            } else if line.starts_with("vn ") {
                normals.push(ObjFile::parse_vec3_line(line.as_str())?)
            } else if line.starts_with("vt ") {
                textures.push(ObjFile::parse_vec2_line(line.as_str())?)
            } else if line.starts_with("f ") {
                if let Some(current_material_index) = current_material_index {
                    indices_materials.push(current_material_index);
                }
                ObjFile::parse_index_line(line.as_str(), &mut indices, current_material_index)?;
            } else if line.starts_with("usemtl ") {
                if let Some(ref materials) = materials {
                    let material_name = &line[7..].trim();
                    let material_index = materials
                        .iter()
                        .position(|m| &m.name.as_str() == material_name)
                        .ok_or(())?;
                    current_material_index = Some(material_index as u8);
                }
            } else if line.starts_with("mtllib ") {
                let filename = &line.as_str()[7..].trim();
                materials = Some(Self::read_mtl_lib(filename)?);
            }
        }
        Ok(ObjFile {
            vertices,
            normals,
            textures,
            indices,
            materials,
        })
    }

    fn parse_index_line(
        line: &str,
        indices: &mut Vec<IndexTuoel>,
        current_material_index: Option<u8>,
    ) -> Result<(), ()> {
        let mut parts = line.split_whitespace();
        parts.next();
        let (vert_1, texture_1, norm_1) = ObjFile::parse_next_index_component(&mut parts)?;
        let (vert_2, texture_2, norm_2) = ObjFile::parse_next_index_component(&mut parts)?;
        let (vert_3, texture_3, norm_3) = ObjFile::parse_next_index_component(&mut parts)?;
        indices.push(IndexTuoel {
            vertex: [vert_1 - 1, vert_2 - 1, vert_3 - 1],
            texture: texture_1.map(|texture_1| {
                [
                    texture_1 - 1,
                    texture_2.unwrap() - 1,
                    texture_3.unwrap() - 1,
                ]
            }),
            normals: [norm_1 - 1, norm_2 - 1, norm_3 - 1],
            style: current_material_index,
        });

        if let Ok((vert_4, texture_4, norm_4)) = ObjFile::parse_next_index_component(&mut parts) {
            indices.push(IndexTuoel {
                vertex: [vert_1 - 1, vert_3 - 1, vert_4 - 1],
                texture: texture_1.map(|texture_1| {
                    [
                        texture_1 - 1,
                        texture_3.unwrap() - 1,
                        texture_4.unwrap() - 1,
                    ]
                }),
                normals: [norm_1 - 1, norm_3 - 1, norm_4 - 1],
                style: current_material_index,
            });
        }

        Ok(())
    }

    fn parse_next_index_component(
        parts: &mut SplitWhitespace,
    ) -> Result<(u16, Option<u16>, u16), ()> {
        let component = parts.next().ok_or(())?;
        let first_divider_pos = component.find("/").ok_or(())?;
        let (vertex, tail) = component.split_at(first_divider_pos);
        let second_divider_pos = tail[1..].find("/").ok_or(())?;
        let (texture, normal) = tail[1..].split_at(second_divider_pos);

        let mut texture_index = None;
        if !texture.is_empty() {
            texture_index = Some(texture.parse().map_err(|_| ()).unwrap())
        }

        Ok((
            vertex.parse().map_err(|_| ())?,
            texture_index,
            normal[1..].parse().map_err(|_| ())?,
        ))
    }

    fn parse_vec2_line(line: &str) -> Result<[f32; 2], ()> {
        let mut parts = line.split_whitespace();
        parts.next();

        Ok([
            ObjFile::parse_next_float_component(&mut parts)?,
            ObjFile::parse_next_float_component(&mut parts)?,
        ])
    }

    fn parse_vec3_line(line: &str) -> Result<[f32; 3], ()> {
        let mut parts = line.split_whitespace();
        parts.next();

        Ok([
            ObjFile::parse_next_float_component(&mut parts)?,
            ObjFile::parse_next_float_component(&mut parts)?,
            ObjFile::parse_next_float_component(&mut parts)?,
        ])
    }

    fn parse_next_float_component(parts: &mut SplitWhitespace) -> Result<f32, ()> {
        Ok(parts.next().ok_or(())?.parse().map_err(|_| ())?)
    }
}

#[derive(Debug)]
pub struct Material {
    name: String,
    pub ambient_color: Option<[f32; 3]>,
    pub diffuse_color: Option<[f32; 3]>,
    pub specular_color: Option<[f32; 3]>,
    pub specular_exponent: Option<u16>,
    pub texture: Option<Image>,
}
