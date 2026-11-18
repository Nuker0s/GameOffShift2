using System;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering.Universal;

/// <summary>
/// URP 2D shadow provider for the union outline of a solidified particle blob.
/// Select "Solidified Particle Outline" as the Casting Source on the
/// ShadowCaster2D component in the configured shadow prefab.
/// </summary>
[Serializable]
public sealed class SolidifiedParticleShadowProvider : ShadowShape2DProvider
{
    public override GUIContent ProviderName(string componentName)
    {
        return new GUIContent("Solidified Particle Outline");
    }

    public override bool IsRequiredComponentData(Component sourceComponent)
    {
        return sourceComponent is SolidifiedBodyShadowData;
    }

    public override void OnBeforeRender(
        Component sourceComponent,
        Bounds worldCullingBounds,
        ShadowShape2D persistentShadowShape)
    {
        SolidifiedBodyShadowData shadowData = sourceComponent as SolidifiedBodyShadowData;

        if (shadowData == null)
            return;

        Vector2[] localVertices = shadowData.BoundaryVertices;
        int[] sourceIndices = shadowData.BoundaryIndices;

        if (localVertices == null || localVertices.Length < 3 ||
            sourceIndices == null || sourceIndices.Length < 6 || sourceIndices.Length % 2 != 0)
        {
            return;
        }

        // Fill managed arrays first. NativeArray variables declared directly in
        // a using statement are readonly, so their indexers cannot be assigned.
        Vector3[] worldVertexArray = new Vector3[localVertices.Length];

        for (int i = 0; i < localVertices.Length; i++)
        {
            Vector2 localPoint = localVertices[i];
            worldVertexArray[i] = shadowData.transform.TransformPoint(
                new Vector3(localPoint.x, localPoint.y, 0f));
        }

        int[] indexArray = new int[sourceIndices.Length];
        Array.Copy(sourceIndices, indexArray, sourceIndices.Length);

        using (NativeArray<Vector3> worldVertices = new NativeArray<Vector3>(worldVertexArray, Allocator.Temp))
        using (NativeArray<int> indices = new NativeArray<int>(indexArray, Allocator.Temp))
        {
            persistentShadowShape.SetShape(
                worldVertices,
                indices,
                ShadowShape2D.OutlineTopology.Lines,
                ShadowShape2D.WindingOrder.Clockwise,
                true,
                true);
        }
    }
}
