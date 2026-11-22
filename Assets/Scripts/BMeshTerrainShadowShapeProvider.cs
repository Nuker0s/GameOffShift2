using System;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering.Universal;

/// <summary>
/// URP ShadowCaster2D source for BMeshTerrain2D.
/// Select "BMesh Terrain" in the ShadowCaster2D Casting Source dropdown.
///
/// BMeshTerrain2D.GeneratedOutline is stored in terrain-local coordinates.
/// ShadowCaster2D applies the GameObject transform itself at render time,
/// so this provider passes those local vertices with Matrix4x4.identity.
/// </summary>
[Serializable]
public sealed class BMeshTerrainShadowShapeProvider : ShadowShape2DProvider
{
    [NonSerialized] private int lastAppliedGeneration = int.MinValue;
    [NonSerialized] private bool hasAppliedShape;

    public override GUIContent ProviderName(string componentName)
    {
        return new GUIContent("BMesh Terrain");
    }

    public override bool IsRequiredComponentData(Component sourceComponent)
    {
        return sourceComponent is BMeshTerrain2D;
    }

    public override void OnInitialized(
        Component sourceComponent,
        ShadowShape2D persistentShadowShape)
    {
        ApplyGeneratedShape(sourceComponent, persistentShadowShape, true);
    }

    public override void OnBeforeRender(
        Component sourceComponent,
        Bounds worldCullingBounds,
        ShadowShape2D persistentShadowShape)
    {
        ApplyGeneratedShape(sourceComponent, persistentShadowShape, false);
    }

    private void ApplyGeneratedShape(
        Component sourceComponent,
        ShadowShape2D shadowShape,
        bool force)
    {
        BMeshTerrain2D terrain = sourceComponent as BMeshTerrain2D;

        if (terrain == null || shadowShape == null)
            return;

        System.Collections.Generic.IReadOnlyList<Vector3> outline = terrain.GeneratedOutline;

        if (outline == null || outline.Count < 3)
            return;

        int currentGeneration = terrain.GenerationVersion;

        if (!force &&
            hasAppliedShape &&
            currentGeneration == lastAppliedGeneration)
        {
            return;
        }

        int pointCount = outline.Count;

        NativeArray<Vector3> vertices = new NativeArray<Vector3>(
            pointCount,
            Allocator.Temp
        );

        NativeArray<int> indices = new NativeArray<int>(
            pointCount * 2,
            Allocator.Temp
        );

        NativeArray<float> radii = new NativeArray<float>(
            pointCount,
            Allocator.Temp
        );

        try
        {
            for (int i = 0; i < pointCount; i++)
            {
                // Keep vertices in local space. Do NOT call TransformPoint here.
                vertices[i] = outline[i];
                radii[i] = 0f;

                indices[i * 2] = i;
                indices[i * 2 + 1] = (i + 1) % pointCount;
            }

            // The caster is on the terrain root, so its transform is the terrain
            // transform. The vertices are already terrain-local.
            shadowShape.SetShape(
                vertices,
                indices,
                radii,
                Matrix4x4.identity,
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

            if (radii.IsCreated)
                radii.Dispose();
        }

        lastAppliedGeneration = currentGeneration;
        hasAppliedShape = true;
    }
}
