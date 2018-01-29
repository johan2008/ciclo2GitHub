
/*varying vec4 ShadowCoord;

varying vec2 texCoord;
attribute  vec2 vTexCoord;


void main()
{

	gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
    gl_TexCoord[0] = gl_MultiTexCoord0;

}
*/

// varying vec3 L; //output vector from vertex to light, in tangent space
// varying vec3 V; //output vector from vertex to eye, in tangent space
attribute vec3 objSpaceTangent; //vertex tangent vector in object (model) space
void main() {
	//output the clip-space vertex position
	gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
	
	//output vertex STPQ texture coordinates
	gl_TexCoord[0] = gl_MultiTexCoord0;
	
	// //get eye-space vertex and light positions
	vec3 eyeSpaceVertexPos = vec3 (gl_ModelViewMatrix * gl_Vertex);
	vec3 eyeSpaceLightPos = vec3 (gl_LightSource[0].position) ;
	// //compute normal, tangent, and binormal vectors in eye space
	vec3 N = normalize(gl_NormalMatrix * gl_Normal);
	vec3 T = normalize(gl_NormalMatrix * objSpaceTangent);
	vec3 B = cross (N, T);
	// //compute light vector L in tangent space
	vec3 eyeSpaceVertexToLight = vec3(eyeSpaceLightPos–eyeSpaceVertexPos);
	L.x = dot (T, eyeSpaceVertexToLight);
	L.y = dot (B, eyeSpaceVertexToLight);
	L.z = dot (N, eyeSpaceVertexToLight);
	L = normalize(L);
	// //compute view vector V in tangent space
	vec3 eyeSpaceVertexToEye = vec3(0,0,0)–eyeSpaceVertexPos;
	V.x = dot (T, eyeSpaceVertexToEye);
	V.y = dot (B, eyeSpaceVertexToEye);
	V.z = dot (N, eyeSpaceVertexToEye);
	V = normalize(V);
}