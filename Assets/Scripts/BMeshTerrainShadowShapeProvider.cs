using System;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering.Universal;

/// <summary>
/// Supplies BMeshTerrain2D's last-generated outline to URP's ShadowCaster2D.
/// Select "BMesh Terrain" in the ShadowCaster2D Casting Source dropdown.
///
/// The provider deliberately calls SetShape on every OnBeforeRender callback.
/// ShadowShape2D is owned by URP; caching only by outline version can leave a
/// newly-created ShadowShape2D instance empty.
/// </summary>
[Serializable]
public sealed class BMeshTerrainShadowShapeProvider : ShadowShape2DProvider
{
    public override GUIContent ProviderName(string componentName)
    {
        return new GUIContent("BMesh Terrain");
    }

    public override bool IsRequiredComponentData(Component currentComponent)
    {
        return currentComponent is BMeshTerrain2D;
    }

    public override void OnInitialized(
        Component sourceComponent,
        ShadowShape2D persistentShadowShape)
    {
        ApplyGeneratedShape(sourceComponent, persistentShadowShape);
    }

    public override void OnBeforeRender(
        Component sourceComponent,
        Bounds worldCullingBounds,
        ShadowShape2D persistentShadowShape)
    {
        ApplyGeneratedShape(sourceComponent, persistentShadowShape);
    }

    private static void ApplyGeneratedShape(
        Component sourceComponent,
        ShadowShape2D shadowShape)
    {
        if (sourceComponent == null || shadowShape == null)
            return;

        // Normally sourceComponent is the BMeshTerrain2D component because
        // IsRequiredComponentData returns true for that component. The fallback
        // also handles URP versions that pass another component on the object.
        BMeshTerrain2D terrain = sourceComponent as BMeshTerrain2D;

        if (terrain == null)
            terrain = sourceComponent.GetComponent<BMeshTerrain2D>();

        if (terrain == null)
            return;

        System.Collections.Generic.IReadOnlyList<Vector3> outline =
            terrain.GeneratedOutline;

        if (outline == null || outline.Count < 3)
            return;

        int pointCount = outline.Count;

        NativeArray<Vector3> vertices = new NativeArray<Vector3>(
            pointCount,
            Allocator.Temp
        );

        NativeArray<int> indices = new NativeArray<int>(
            pointCount * 2,
            Allocator.Temp
        );

        try
        {
            for (int i = 0; i < pointCount; i++)
            {
                // ShadowShape2D expects world-space vertices.
                vertices[i] = terrain.transform.TransformPoint(outline[i]);

                // Lines topology uses pairs of vertex indices to describe
                // each edge of the closed outline.
                indices[i * 2] = i;
                indices[i * 2 + 1] = (i + 1) % pointCount;
            }

            // BMeshTerrain2D normalizes its generated outline to CCW winding.
            // Interior geometry is required because this source is not a Renderer.
            shadowShape.SetShape(
                vertices,
                indices,
                ShadowShape2D.OutlineTopology.Lines,
                ShadowShape2D.WindingOrder.CounterClockwise,
                true,
                true
            );
        }
        finally
        {
            if (vertices.IsCreated)
                vertices.Dispose();

            if (indices.IsCreated)
                indices.Dispose();
        }
    }
}
