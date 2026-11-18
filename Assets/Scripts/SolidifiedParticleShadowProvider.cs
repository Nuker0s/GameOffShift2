using System;
using System.Collections.Generic;
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

        // SetShape(OutlineTopology.Lines) with interior geometry expects each
        // contour to be supplied as an ordered, closed chain of line pairs.
        // Refuse malformed input before it reaches URP's Burst tessellator.
        if (!HasOnlyClosedContours(sourceIndices, localVertices.Length))
        {
            Debug.LogError(
                "[SolidifiedParticleShadowProvider] Shadow outline contains an open, " +
                "disconnected, or malformed contour. Skipping this shadow update.", shadowData);
            return;
        }

        NativeArray<Vector3> worldVertices = default;
        NativeArray<int> indices = default;

        try
        {
            worldVertices = new NativeArray<Vector3>(localVertices.Length, Allocator.Temp);
            indices = new NativeArray<int>(sourceIndices.Length, Allocator.Temp);

            for (int i = 0; i < localVertices.Length; i++)
            {
                Vector2 localPoint = localVertices[i];
                worldVertices[i] = shadowData.transform.TransformPoint(
                    new Vector3(localPoint.x, localPoint.y, 0f));
            }

            for (int i = 0; i < sourceIndices.Length; i++)
                indices[i] = sourceIndices[i];

            persistentShadowShape.SetShape(
                worldVertices,
                indices,
                ShadowShape2D.OutlineTopology.Lines,
                ShadowShape2D.WindingOrder.Clockwise,
                true,
                true);
        }
        finally
        {
            if (worldVertices.IsCreated)
                worldVertices.Dispose();

            if (indices.IsCreated)
                indices.Dispose();
        }
    }

    private static bool HasOnlyClosedContours(int[] indices, int vertexCount)
    {
        int contourStart = -1;
        int previousEnd = -1;
        int contourEdgeCount = 0;
        bool contourClosed = false;
        HashSet<int> contourVertices = new HashSet<int>();

        for (int i = 0; i < indices.Length; i += 2)
        {
            int start = indices[i];
            int end = indices[i + 1];

            if (start < 0 || start >= vertexCount || end < 0 || end >= vertexCount || start == end)
                return false;

            if (contourStart < 0 || contourClosed)
            {
                if (contourStart >= 0 && (!contourClosed || contourEdgeCount < 3))
                    return false;

                if (contourStart >= 0 && start == previousEnd)
                    return false;

                contourStart = start;
                previousEnd = start;
                contourEdgeCount = 0;
                contourClosed = false;
                contourVertices.Clear();
                contourVertices.Add(start);
            }
            else if (start != previousEnd)
            {
                return false;
            }

            if (end == contourStart)
            {
                contourEdgeCount++;

                if (contourEdgeCount < 3)
                    return false;

                contourClosed = true;
            }
            else
            {
                if (contourVertices.Contains(end))
                    return false;

                contourVertices.Add(end);
                contourEdgeCount++;
            }

            previousEnd = end;
        }

        return contourClosed && contourEdgeCount >= 3;
    }
}
