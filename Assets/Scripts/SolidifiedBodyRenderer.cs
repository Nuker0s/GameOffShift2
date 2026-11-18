using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using Unity.U2D.Physics;

/// <summary>
/// Builds and follows the visual mesh for one ParticleSolidifier compound body.
/// The mesh is generated in the physics body's local space and the GameObject
/// follows the PhysicsBody pose.
/// </summary>
[DisallowMultipleComponent]
public sealed class SolidifiedBodyRenderer : MonoBehaviour
{
    private PhysicsBody solidBody;
    private Mesh ownedMesh;
    private float renderZOffset;
    private bool initialized;

    public static SolidifiedBodyRenderer Create(
        PhysicsBody body,
        ParticleSolidifierManifest manifest,
        SolidifiedParticleRenderConfig config)
    {
        if (!body.isValid)
            throw new System.ArgumentException("Cannot render an invalid PhysicsBody.", nameof(body));

        if (manifest == null)
            throw new System.ArgumentNullException(nameof(manifest));

        if (config == null)
            throw new System.ArgumentNullException(nameof(config));

        if (config.RenderMaterial == null)
        {
            Debug.LogWarning(
                "[SolidifiedBodyRenderer] No render material is assigned to SolidifiedParticleRenderConfig. " +
                "The solidified body will not have a visual mesh.", config);
            return null;
        }

        GameObject visualObject = new GameObject("Solidified Body Visual");
        visualObject.layer = config.GameObjectLayer;

        SolidifiedBodyRenderer renderer = visualObject.AddComponent<SolidifiedBodyRenderer>();

        try
        {
            renderer.Initialize(body, manifest, config);

            if (!renderer.initialized)
            {
                if (Application.isPlaying)
                    UnityEngine.Object.Destroy(visualObject);
                else
                    UnityEngine.Object.DestroyImmediate(visualObject);

                return null;
            }

            return renderer;
        }
        catch
        {
            if (Application.isPlaying)
                UnityEngine.Object.Destroy(visualObject);
            else
                UnityEngine.Object.DestroyImmediate(visualObject);

            throw;
        }
    }

    private void Initialize(
        PhysicsBody body,
        ParticleSolidifierManifest manifest,
        SolidifiedParticleRenderConfig config)
    {
        solidBody = body;
        renderZOffset = config.RenderZOffset;

        ownedMesh = BuildMesh(manifest.snapshots);

        if (ownedMesh == null || ownedMesh.vertexCount == 0)
        {
            Debug.LogWarning(
                "[SolidifiedBodyRenderer] The solidified body has no visible polygon geometry. " +
                "This can happen when terrain subtraction removes the entire blob.", this);
            return;
        }

        ownedMesh.name = "Solidified Particle Mesh";

        MeshFilter meshFilter = gameObject.AddComponent<MeshFilter>();
        meshFilter.sharedMesh = ownedMesh;

        MeshRenderer meshRenderer = gameObject.AddComponent<MeshRenderer>();
        meshRenderer.sharedMaterial = config.RenderMaterial;
        meshRenderer.sortingLayerID = SortingLayer.NameToID(config.SortingLayerName);
        meshRenderer.sortingOrder = config.SortingOrder;
        meshRenderer.shadowCastingMode = ShadowCastingMode.Off;
        meshRenderer.receiveShadows = false;

        if (config.ShadowCasterPrefab != null)
            CreateShadowCaster(config.ShadowCasterPrefab, manifest.snapshots);

        initialized = true;
        SyncTransform();
    }

    private void LateUpdate()
    {
        if (!initialized)
            return;

        if (!solidBody.isValid)
        {
            Destroy(gameObject);
            return;
        }

        SyncTransform();
    }

    private void SyncTransform()
    {
        Vector2 bodyPosition = solidBody.position;
        float bodyAngle = solidBody.rotation.degrees;

        transform.SetPositionAndRotation(
            new Vector3(bodyPosition.x, bodyPosition.y, renderZOffset),
            Quaternion.Euler(0f, 0f, bodyAngle));
    }

    private static Mesh BuildMesh(List<ParticleSolidifierSnapshot> snapshots)
    {
        List<Vector3> vertices = new List<Vector3>();
        List<Color> colors = new List<Color>();
        List<Vector3> normals = new List<Vector3>();
        List<int> triangles = new List<int>();

        if (snapshots == null)
            return null;

        for (int snapshotIndex = 0; snapshotIndex < snapshots.Count; snapshotIndex++)
        {
            ParticleSolidifierSnapshot snapshot = snapshots[snapshotIndex];

            if (snapshot == null || snapshot.renderPolygons == null)
                continue;

            for (int polygonIndex = 0; polygonIndex < snapshot.renderPolygons.Count; polygonIndex++)
            {
                Vector2[] polygon = snapshot.renderPolygons[polygonIndex];

                if (polygon == null || polygon.Length < 3)
                    continue;

                int firstVertex = vertices.Count;
                float signedArea = CalculateSignedArea(polygon);

                for (int vertexIndex = 0; vertexIndex < polygon.Length; vertexIndex++)
                {
                    vertices.Add(new Vector3(polygon[vertexIndex].x, polygon[vertexIndex].y, 0f));
                    colors.Add(snapshot.particleColor);
                    normals.Add(Vector3.back);
                }

                // Each PhysicsComposer output is convex, so a triangle fan is
                // sufficient. Reverse the fan as needed so every triangle faces
                // the usual 2D camera direction (negative Z).
                for (int triangleIndex = 1; triangleIndex < polygon.Length - 1; triangleIndex++)
                {
                    triangles.Add(firstVertex);

                    if (signedArea >= 0f)
                    {
                        triangles.Add(firstVertex + triangleIndex + 1);
                        triangles.Add(firstVertex + triangleIndex);
                    }
                    else
                    {
                        triangles.Add(firstVertex + triangleIndex);
                        triangles.Add(firstVertex + triangleIndex + 1);
                    }
                }
            }
        }

        if (vertices.Count == 0)
            return null;

        Mesh mesh = new Mesh();
        mesh.indexFormat = vertices.Count > 65535
            ? IndexFormat.UInt32
            : IndexFormat.UInt16;

        mesh.SetVertices(vertices);
        mesh.SetColors(colors);
        mesh.SetNormals(normals);
        mesh.SetTriangles(triangles, 0, true);
        mesh.RecalculateBounds();

        return mesh;
    }

    private const float OutlineMergeTolerance = 0.0005f;

    private struct OutlineSegment
    {
        public Vector2 start;
        public Vector2 end;

        public OutlineSegment(Vector2 start, Vector2 end)
        {
            this.start = start;
            this.end = end;
        }
    }

    private struct OutlineSplitPoint
    {
        public float t;
        public Vector2 point;

        public OutlineSplitPoint(float t, Vector2 point)
        {
            this.t = t;
            this.point = point;
        }
    }

    private struct OutlinePointKey : IEquatable<OutlinePointKey>, IComparable<OutlinePointKey>
    {
        public long x;
        public long y;

        public OutlinePointKey(Vector2 point)
        {
            x = (long)Math.Round(point.x / OutlineMergeTolerance);
            y = (long)Math.Round(point.y / OutlineMergeTolerance);
        }

        public bool Equals(OutlinePointKey other)
        {
            return x == other.x && y == other.y;
        }

        public override bool Equals(object obj)
        {
            return obj is OutlinePointKey other && Equals(other);
        }

        public override int GetHashCode()
        {
            unchecked
            {
                return (x.GetHashCode() * 397) ^ y.GetHashCode();
            }
        }

        public int CompareTo(OutlinePointKey other)
        {
            int xComparison = x.CompareTo(other.x);
            return xComparison != 0 ? xComparison : y.CompareTo(other.y);
        }
    }

    private struct OutlineEdgeKey : IEquatable<OutlineEdgeKey>
    {
        private readonly OutlinePointKey first;
        private readonly OutlinePointKey second;

        public OutlineEdgeKey(OutlinePointKey a, OutlinePointKey b)
        {
            if (a.CompareTo(b) <= 0)
            {
                first = a;
                second = b;
            }
            else
            {
                first = b;
                second = a;
            }
        }

        public bool Equals(OutlineEdgeKey other)
        {
            return first.Equals(other.first) && second.Equals(other.second);
        }

        public override bool Equals(object obj)
        {
            return obj is OutlineEdgeKey other && Equals(other);
        }

        public override int GetHashCode()
        {
            unchecked
            {
                return (first.GetHashCode() * 397) ^ second.GetHashCode();
            }
        }
    }

    private void CreateShadowCaster(
        GameObject shadowCasterPrefab,
        List<ParticleSolidifierSnapshot> snapshots)
    {
        Vector2[] boundaryVertices;
        int[] boundaryIndices;
        BuildBoundaryEdgeData(snapshots, out boundaryVertices, out boundaryIndices);

        if (boundaryVertices.Length < 3 || boundaryIndices.Length < 6)
        {
            Debug.LogWarning(
                "[SolidifiedBodyRenderer] Shadow prefab was assigned, but no outer contour could be generated.", this);
            return;
        }

        GameObject shadowObject = Instantiate(shadowCasterPrefab, transform, false);
        shadowObject.name = "Solidified Body 2D Shadow";
        shadowObject.transform.localPosition = Vector3.zero;
        shadowObject.transform.localRotation = Quaternion.identity;
        shadowObject.transform.localScale = Vector3.one;

        SolidifiedBodyShadowData shadowData = shadowObject.GetComponent<SolidifiedBodyShadowData>();
        ShadowCaster2D shadowCaster = shadowObject.GetComponent<ShadowCaster2D>();

        if (shadowData == null || shadowCaster == null)
        {
            Debug.LogError(
                "[SolidifiedBodyRenderer] The shadow prefab must contain both SolidifiedBodyShadowData " +
                "and ShadowCaster2D components. The shadow object was removed.", shadowObject);
            Destroy(shadowObject);
            return;
        }

        shadowData.SetBoundaryShape(boundaryVertices, boundaryIndices);
        shadowCaster.castsShadows = true;
    }

    /// <summary>
    /// Extracts the union boundary from all particle polygons. Shared internal
    /// edges are cancelled. Long edges are split at other polygon endpoints
    /// first, which also handles T-junctions caused by polygon fragmentation.
    /// Returned indices describe connected line segments, sharing vertices at
    /// contour corners so the shadow provider receives closed loops.
    /// </summary>
    private static void BuildBoundaryEdgeData(
        List<ParticleSolidifierSnapshot> snapshots,
        out Vector2[] boundaryVertices,
        out int[] boundaryIndices)
    {
        boundaryVertices = new Vector2[0];
        boundaryIndices = new int[0];

        List<OutlineSegment> segments = new List<OutlineSegment>();
        Dictionary<OutlinePointKey, Vector2> uniquePointLookup =
            new Dictionary<OutlinePointKey, Vector2>();

        if (snapshots == null)
            return;

        for (int snapshotIndex = 0; snapshotIndex < snapshots.Count; snapshotIndex++)
        {
            ParticleSolidifierSnapshot snapshot = snapshots[snapshotIndex];

            if (snapshot == null || snapshot.renderPolygons == null)
                continue;

            for (int polygonIndex = 0; polygonIndex < snapshot.renderPolygons.Count; polygonIndex++)
            {
                Vector2[] polygon = snapshot.renderPolygons[polygonIndex];

                if (polygon == null || polygon.Length < 3)
                    continue;

                for (int vertexIndex = 0; vertexIndex < polygon.Length; vertexIndex++)
                {
                    Vector2 sourceA = polygon[vertexIndex];
                    Vector2 sourceB = polygon[(vertexIndex + 1) % polygon.Length];

                    OutlinePointKey keyA = new OutlinePointKey(sourceA);
                    OutlinePointKey keyB = new OutlinePointKey(sourceB);

                    if (keyA.Equals(keyB))
                        continue;

                    Vector2 pointA = GetOrAddOutlinePoint(keyA, sourceA, uniquePointLookup);
                    Vector2 pointB = GetOrAddOutlinePoint(keyB, sourceB, uniquePointLookup);
                    segments.Add(new OutlineSegment(pointA, pointB));
                }
            }
        }

        if (segments.Count == 0)
            return;

        List<Vector2> allPoints = new List<Vector2>(uniquePointLookup.Values);
        allPoints.Sort((a, b) => a.x.CompareTo(b.x));

        Dictionary<OutlineEdgeKey, int> edgeCounts = new Dictionary<OutlineEdgeKey, int>();
        Dictionary<OutlineEdgeKey, OutlineSegment> uniqueEdges =
            new Dictionary<OutlineEdgeKey, OutlineSegment>();

        float toleranceSquared = OutlineMergeTolerance * OutlineMergeTolerance;

        for (int segmentIndex = 0; segmentIndex < segments.Count; segmentIndex++)
        {
            OutlineSegment segment = segments[segmentIndex];
            Vector2 direction = segment.end - segment.start;
            float lengthSquared = direction.sqrMagnitude;

            if (lengthSquared <= toleranceSquared)
                continue;

            List<OutlineSplitPoint> splitPoints = new List<OutlineSplitPoint>();
            splitPoints.Add(new OutlineSplitPoint(0f, segment.start));
            splitPoints.Add(new OutlineSplitPoint(1f, segment.end));

            float minimumX = Mathf.Min(segment.start.x, segment.end.x) - OutlineMergeTolerance;
            float maximumX = Mathf.Max(segment.start.x, segment.end.x) + OutlineMergeTolerance;
            int firstCandidate = FindFirstPointWithXAtLeast(allPoints, minimumX);

            for (int pointIndex = firstCandidate;
                pointIndex < allPoints.Count && allPoints[pointIndex].x <= maximumX;
                pointIndex++)
            {
                Vector2 point = allPoints[pointIndex];
                float t = Vector2.Dot(point - segment.start, direction) / lengthSquared;

                if (t <= 0.00001f || t >= 0.99999f)
                    continue;

                Vector2 closestPoint = segment.start + direction * t;

                if ((point - closestPoint).sqrMagnitude <= toleranceSquared)
                    splitPoints.Add(new OutlineSplitPoint(t, point));
            }

            splitPoints.Sort((a, b) => a.t.CompareTo(b.t));

            for (int splitIndex = 0; splitIndex < splitPoints.Count - 1; splitIndex++)
            {
                Vector2 edgeStart = splitPoints[splitIndex].point;
                Vector2 edgeEnd = splitPoints[splitIndex + 1].point;
                OutlinePointKey startKey = new OutlinePointKey(edgeStart);
                OutlinePointKey endKey = new OutlinePointKey(edgeEnd);

                if (startKey.Equals(endKey))
                    continue;

                OutlineEdgeKey edgeKey = new OutlineEdgeKey(startKey, endKey);

                if (edgeCounts.TryGetValue(edgeKey, out int currentCount))
                {
                    edgeCounts[edgeKey] = currentCount + 1;
                }
                else
                {
                    edgeCounts.Add(edgeKey, 1);
                    uniqueEdges.Add(edgeKey, new OutlineSegment(edgeStart, edgeEnd));
                }
            }
        }

        List<Vector2> resultVertices = new List<Vector2>();
        List<int> resultIndices = new List<int>();
        Dictionary<OutlinePointKey, int> vertexIndexLookup = new Dictionary<OutlinePointKey, int>();

        foreach (KeyValuePair<OutlineEdgeKey, int> edgeEntry in edgeCounts)
        {
            if (edgeEntry.Value != 1)
                continue;

            OutlineSegment edge = uniqueEdges[edgeEntry.Key];
            int startIndex = GetOrAddBoundaryVertex(edge.start, resultVertices, vertexIndexLookup);
            int endIndex = GetOrAddBoundaryVertex(edge.end, resultVertices, vertexIndexLookup);

            if (startIndex == endIndex)
                continue;

            resultIndices.Add(startIndex);
            resultIndices.Add(endIndex);
        }

        boundaryVertices = resultVertices.ToArray();
        boundaryIndices = resultIndices.ToArray();
    }

    private static int GetOrAddBoundaryVertex(
        Vector2 point,
        List<Vector2> vertices,
        Dictionary<OutlinePointKey, int> vertexIndexLookup)
    {
        OutlinePointKey key = new OutlinePointKey(point);

        if (vertexIndexLookup.TryGetValue(key, out int existingIndex))
            return existingIndex;

        int newIndex = vertices.Count;
        vertices.Add(point);
        vertexIndexLookup.Add(key, newIndex);
        return newIndex;
    }

    private static int FindFirstPointWithXAtLeast(List<Vector2> points, float minimumX)
    {
        int low = 0;
        int high = points.Count;

        while (low < high)
        {
            int middle = low + (high - low) / 2;

            if (points[middle].x < minimumX)
                low = middle + 1;
            else
                high = middle;
        }

        return low;
    }

    private static Vector2 GetOrAddOutlinePoint(
        OutlinePointKey key,
        Vector2 point,
        Dictionary<OutlinePointKey, Vector2> pointLookup)
    {
        if (pointLookup.TryGetValue(key, out Vector2 existingPoint))
            return existingPoint;

        pointLookup.Add(key, point);
        return point;
    }

    private static float CalculateSignedArea(Vector2[] polygon)
    {
        float twiceArea = 0f;

        for (int i = 0; i < polygon.Length; i++)
        {
            Vector2 current = polygon[i];
            Vector2 next = polygon[(i + 1) % polygon.Length];
            twiceArea += current.x * next.y - next.x * current.y;
        }

        return twiceArea * 0.5f;
    }

    private void OnDestroy()
    {
        if (ownedMesh == null)
            return;

        if (Application.isPlaying)
            UnityEngine.Object.Destroy(ownedMesh);
        else
            UnityEngine.Object.DestroyImmediate(ownedMesh);

        ownedMesh = null;
    }
}
