

#version 120  // YJC: Comment/un-comment this line to resolve compilation errors
                 //      due to different settings of the default GLSL version

//in  vec4 color;
//out vec4 fColor;


varying vec4 color;
//varying vec4 fColor;


varying  vec3 fN;
//varying  vec3 fL;
varying  vec3 fE;




uniform vec3 objectColor;
uniform vec3 lightColor;
uniform vec3 lightPos;
uniform vec3 viewPos;


uniform vec3 fL;//LIGHT DIRECTIONAL


uniform vec4 AmbientProduct, DiffuseProduct, SpecularProduct;
uniform float Shininess;
uniform vec4 LightPosition;


uniform float flag;

uniform float flagS;


varying vec3 Normal; 

varying vec3 FragPos; 


uniform vec4 CameraEye;
uniform vec4 FogColor;

varying vec4 vertex;

/*    
float getFogFactor(float d)
{
    const float FogMax = 20.0;
    const float FogMin = 10.0;

    if (d>=FogMax) return 1;
    if (d<=FogMin) return 0;

    return 1 - (FogMax - d) / (FogMax - FogMin);
}
*/



void main() 
{ 


	if(flag>1){

        vec3 Nor = normalize(fN);   
        vec3 E = normalize(fE);  
        vec3 L = normalize(fL); //luz directional

        vec3 H = normalize( L + E );
        
        vec4 ambient = AmbientProduct;

        float Kd = max(dot(L, Nor), 0.0);
        vec4 diffuse = Kd * DiffuseProduct;
        
        float Ks = pow(max(dot(Nor, H), 0.0), Shininess);
        vec4 specular = Ks*SpecularProduct;

        // discard the specular highlight if the light's behind the vertex
        if( dot(L, Nor) < 0.0 ) {
    		specular = vec4(0.0, 0.0, 0.0, 1.0);
        }

        gl_FragColor = (ambient + diffuse + specular);
        //gl_FragColor = (ambient + diffuse );


        gl_FragColor.a = 1.0;


        //float d = distance(CameraEye, V);
        float d =  sqrt( pow(CameraEye.x-vertex.x,2) + pow(CameraEye.y-vertex.y,2) + pow(CameraEye.z-vertex.z,2) + pow(CameraEye.w-vertex.w,2)  );
        //float d = (CameraEye -eye)
        float FogMax = 18.0;
        float FogMin = 0.0;

        //if (d>=FogMax) return 1;
        //  
        //if (d<=FogMin) return 0;

        float alpha ;//= 1 - (FogMax - d) / (FogMax - FogMin);//getFogFactor(d);
        if (d>=FogMax) alpha =  1;
        else if(d<=FogMin) alpha = 0;

        else alpha = 1 - (FogMax - d) / (FogMax - FogMin);//getFogFactor(d);
        gl_FragColor = mix((ambient + diffuse + specular)*color, FogColor, alpha);


        /* for light source 2
        L = normalize( pointLightPosition.xyz - pos );
        if(point_light == 0){
            Lf = normalize(pointLightDir.xyz);
        }
        H = normalize( L +  E );
        // get distance attenuation
        dis = length(pos-pointLightPosition.xyz);
        attenuation = 1/(ConstAtt + LinearAtt*dis + QuadAtt*dis*dis);
        if(point_light == 0){   
        //attenuation *= pow(1,pointLightExp);
            if(dot(-L,Lf)>cos(pointLightAng)){
                attenuation *= pow(dot(-L,Lf),pointLightExp);
            }
            else{
                attenuation = 0;
            }
        }

        ambient = attenuation * pointAmbientProduct;
        d = max( dot(L, N), 0.0 );
        diffuse = attenuation * d * pointDiffuseProduct;
        // get specular light
        // pow(x, y) result is undefined if x<0 or if x=0 and y<=0
        s = pow( max(dot(N, H), 0.0), Shininess );
    
        specular = attenuation*s*pointSpecularProduct;
        if( dot(L, N) < 0.0 ) {
            specular = vec4(0.0, 0.0, 0.0, 1.0);
        } 
    */





	}
	else{
	//gl_FragColor = vec4(result,1.0);



    	gl_FragColor = color;

	}	
} 

