using System;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering.Universal;

/// <summary>
/// URP ShadowCaster2D source for BMeshTerrain2D.
///
/// ShadowCaster2D lives on a child GameObject named "ShadowCaster", while
/// BMeshTerrain2D and its MeshRenderer live on the parent terrain GameObject.
/// The provider uses the caster child's Transform as its source component and
/// finds BMeshTerrain2D in the parent hierarchy.
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
        if (sourceComponent is BMeshTerrain2D)
            return true;

        Transform sourceTransform = sourceComponent as Transform;

        if (sourceTransform == null)
            return false;

        return sourceTransform.GetComponentInParent<BMeshTerrain2D>() != null;
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
        if (sourceComponent == null || shadowShape == null)
            return;

        BMeshTerrain2D terrain = sourceComponent as BMeshTerrain2D;

        if (terrain == null)
            terrain = sourceComponent.GetComponentInParent<BMeshTerrain2D>();

        if (terrain == null)
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

        Transform terrainTransform = terrain.transform;
        Transform casterTransform = sourceComponent.transform;

        try
        {
            for (int i = 0; i < pointCount; i++)
            {
                // GeneratedOutline is terrain-local. Convert through world space
                // so the points are also correct in the child caster's local space.
                Vector3 worldPoint = terrainTransform.TransformPoint(outline[i]);
                vertices[i] = casterTransform.InverseTransformPoint(worldPoint);
                radii[i] = 0f;

                // Line segments around the outline. No triangulated mesh is passed
                // into ShadowShape2D by this provider.
                indices[i * 2] = i;
                indices[i * 2 + 1] = (i + 1) % pointCount;
            }

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
