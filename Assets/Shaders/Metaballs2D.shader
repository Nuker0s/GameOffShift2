Shader "Custom/Metaballs2D"
{
    Properties
    {
        _TypeColor0 ("Type 0 Color", Color) = (0.2, 0.6, 1.0, 1.0)
        _TypeColor1 ("Type 1 Color", Color) = (1.0, 0.35, 0.3, 1.0)
        _TypeColor2 ("Type 2 Color", Color) = (0.35, 1.0, 0.5, 1.0)
        _TypeColor3 ("Type 3 Color", Color) = (1.0, 0.85, 0.25, 1.0)
        _Radius ("Metaball Radius", Float) = 0.4
        _Threshold ("Threshold", Float) = 1.0
        _Smoothness ("Edge Smoothness", Float) = 0.2
        _EdgeColor ("Edge Highlight Color", Color) = (1, 1, 1, 1)
        _EdgeWidth ("Edge Highlight Width", Range(0, 1)) = 0.25
        _BackgroundAlpha ("Background Alpha (debug quad visibility)", Range(0, 1)) = 0.08
        _BackgroundColor ("Background Color", Color) = (0.5, 0.5, 0.5, 1.0)
    }

    SubShader
    {
        Tags
        {
            "RenderType" = "Transparent"
            "Queue" = "Transparent"
            "RenderPipeline" = "UniversalPipeline"
            "IgnoreProjector" = "True"
            "PreviewType" = "Plane"
        }

        // Shared code for both passes (2D renderer uses Universal2D,
        // 3D/Forward renderer uses UniversalForward).
        HLSLINCLUDE
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct Attributes
            {
                float4 positionOS : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float3 worldPos : TEXCOORD0;
                float2 uv : TEXCOORD1;
            };

            CBUFFER_START(UnityPerMaterial)
                float4 _TypeColor0;
                float4 _TypeColor1;
                float4 _TypeColor2;
                float4 _TypeColor3;
                float _Radius;
                float _Threshold;
                float _Smoothness;
                float4 _EdgeColor;
                float _EdgeWidth;
                float _BackgroundAlpha;
                float4 _BackgroundColor;
                int _MetaballCount;
            CBUFFER_END

            // xy = world XY position, z = particle type id (float, 0..3).
            StructuredBuffer<float3> _Metaballs;

            Varyings vert (Attributes IN)
            {
                Varyings OUT;
                float3 wpos = mul(unity_ObjectToWorld, IN.positionOS).xyz;
                OUT.worldPos = wpos;
                OUT.uv = IN.uv;
                OUT.positionHCS = TransformWorldToHClip(wpos);
                return OUT;
            }

            half4 frag (Varyings IN) : SV_Target
            {
                float2 wp = IN.worldPos.xy;

                float r2 = max(_Radius * _Radius, 1e-5);
                // Skip far-away particles early (perf). Field beyond this is negligible.
                float maxDist2 = r2 * 16.0; // 4x radius

                float f0 = 0.0;
                float f1 = 0.0;
                float f2 = 0.0;
                float f3 = 0.0;

                // NOTE: cost is O(pixels * count). Fine for a few hundred
                // particles on a single quad. For thousands, consider a
                // render-texture / splat based approach instead.
                for (int i = 0; i < _MetaballCount; ++i)
                {
                    float3 p = _Metaballs[i];
                    float2 d = wp - p.xy;
                    float d2 = dot(d, d);
                    if (d2 > maxDist2)
                        continue;
                    float f = r2 / max(d2, 1e-4);
                    int t = (int)(p.z + 0.5);
                    if (t <= 0)      f0 += f;
                    else if (t == 1) f1 += f;
                    else if (t == 2) f2 += f;
                    else             f3 += f;
                }

                float total = f0 + f1 + f2 + f3;

                float lo = _Threshold - _Smoothness;
                float hi = _Threshold + _Smoothness;
                float alpha = smoothstep(lo, hi, total);

                float invTotal = 1.0 / max(total, 1e-4);
                float3 col =
                    _TypeColor0.rgb * (f0 * invTotal) +
                    _TypeColor1.rgb * (f1 * invTotal) +
                    _TypeColor2.rgb * (f2 * invTotal) +
                    _TypeColor3.rgb * (f3 * invTotal);

                // Bright rim near the threshold -> helps read the gooey edge.
                float edgeBand = 1.0 - smoothstep(_Threshold, _Threshold + _EdgeWidth + 1e-4, total);
                col = lerp(col, _EdgeColor.rgb, edgeBand * _EdgeColor.a * step(_Threshold, total + _Smoothness));

                // If no field here, fall back to faint background so you can
                // still see the quad (proves the mesh/material/camera are fine).
                // Set _BackgroundAlpha to 0 for fully invisible background.
                col = lerp(_BackgroundColor.rgb, col, alpha);
                alpha = max(alpha, _BackgroundAlpha);

                return half4(col, alpha);
            }
        ENDHLSL

        // URP 2D Renderer ONLY draws passes tagged Universal2D.
        // This was the reason for "nothing showing".
        Pass
        {
            Tags { "LightMode" = "Universal2D" }
            Blend SrcAlpha OneMinusSrcAlpha
            ZWrite Off
            Cull Off
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 4.5
            ENDHLSL
        }

        // Universal Forward / Forward+ renderer draws this one.
        Pass
        {
            Tags { "LightMode" = "UniversalForward" }
            Blend SrcAlpha OneMinusSrcAlpha
            ZWrite Off
            Cull Off
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 4.5
            ENDHLSL
        }
    }

    FallBack Off
}

