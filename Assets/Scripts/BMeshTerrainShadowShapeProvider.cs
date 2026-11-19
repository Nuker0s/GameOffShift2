using System;
using System.Collections.Generic;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering.Universal;

/// <summary>
/// URP ShadowCaster2D source for BMeshTerrain2D.
/// This is a serializable provider object owned by ShadowCaster2D, not a
/// MonoBehaviour. Select "BMesh Terrain" in ShadowCaster2D's Casting Source.
/// </summary>
[Serializable]
public sealed class BMeshTerrainShadowShapeProvider : ShadowShape2DProvider
{
    [NonSerialized] private int lastAppliedGeneration = int.MinValue;
    [NonSerialized] private Matrix4x4 lastAppliedTransform;
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

    // This legacy overload is still supported by current URP and is called
    // alongside the Camera overload. It also keeps the provider compatible
    // with the URP versions shipped with Unity 6.5.
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

        IReadOnlyList<Vector3> outline = terrain.GeneratedOutline;

        if (outline == null || outline.Count < 3)
            return;

        Matrix4x4 currentTransform = terrain.transform.localToWorldMatrix;
        int currentGeneration = terrain.GenerationVersion;

        if (!force &&
            hasAppliedShape &&
            currentGeneration == lastAppliedGeneration &&
            currentTransform.Equals(lastAppliedTransform))
        {
            return;
        }

        int pointCount = outline.Count;

        // using variables are readonly in C#, so NativeArray elements cannot
        // be assigned through them. Allocate normally and dispose explicitly.
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
                vertices[i] = terrain.transform.TransformPoint(outline[i]);
                indices[i * 2] = i;
                indices[i * 2 + 1] = (i + 1) % pointCount;
            }

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

        lastAppliedGeneration = currentGeneration;
        lastAppliedTransform = currentTransform;
        hasAppliedShape = true;
    }
}
