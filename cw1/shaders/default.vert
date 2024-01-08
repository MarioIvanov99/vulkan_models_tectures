#version 450 

layout( location = 0 ) in vec3 iPosition; 
layout( location = 1 ) in vec3 iColor; 

layout( set = 0, binding = 0 ) uniform UScene 
{ 
	mat4 camera; 
	mat4 projection;
	mat4 projCam; 
} uScene; 

layout( location = 0 ) out vec3 v2fColor; 

void main() 
{ 
	v2fColor = iColor; 

	gl_Position = uScene.projCam * vec4( iPosition, 1.f ); 

}