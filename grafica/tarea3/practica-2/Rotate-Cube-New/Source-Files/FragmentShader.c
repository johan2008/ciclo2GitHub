/*uniform sampler2D ShadowMap;

varying vec4 ShadowCoord;


varying float texture_mapped_ground;

varying  vec2 texCoord;
uniform sampler2D texMap;


void main()
{	


  	vec4 texColor = texture2D( texMap, gl_TexCoord[0].st );
    gl_FragColor = texColor ;
  
}
*/

varying vec3 L ; //vector from fragment to light, in tangent space
varying vec3 V; //vector from fragment to eye, in tangent space
uniform sampler2D texMap; //texture to be applied to fragment
uniform sampler2D normalMap; //normal vectors for fragments
void main () {
	// //get the normal from the normal map
	vec3 normal = texture2D( normalMap, gl_TexCoord[0].st );
	
	// //unpack the normal (we don’t have a texture combiner to do this for us).
	// //Since normals are packed using (N+1)/2, we apply the inverse of that to unpack
	vec3 N = normalize (2.0 * normal.xyz) – 1.0 ;
	
	// //insure the input light vector is normalized
	vec3 LL = normalize (L);
	
	// //compute Lambertian diffuse coefficient from the normal and light vectors
	float Kd = max( dot(N,LL), 0.0 );
	
	//get texel color from texture map
	vec4 texColor = texture2D( texMap, gl_TexCoord[0].st );
	
	//output the texture color modulated by the diffuse coefficient
	// gl_FragColor = Kd * texColor ;
	gl_FragColor = texColor ;
	// gl_FragColor = vec4(1.0, 0, 0, 1);
}