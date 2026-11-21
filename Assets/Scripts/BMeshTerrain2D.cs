
using System.Collections.Generic;
using Unity.Collections;
using UnityEngine;
using Unity.U2D.Physics;
using UnityEngine.Rendering.Universal;
using static BMesh;

[ExecuteAlways]
[DisallowMultipleComponent]
[RequireComponent(typeof(PhysicsPose))]
[RequireComponent(typeof(MeshFilter))]
[RequireComponent(typeof(MeshRenderer))]
public class BMeshTerrain2D : MonoBehaviour
{
    [Header("Rendering")]
    [SerializeField] private Material material;
    [SerializeField] private Color vertexColor = Color.white;

    [Header("PhysicsCore2D")]
    [SerializeField] private PhysicsShapeDefinition physicsShapeDefinition =
        PhysicsShapeDefinition.defaultDefinition;

    [SerializeField] private string physicsLayerName = "Default";

    [Header("Shadow")]
    [SerializeField] private bool generateShadowCaster = true;
    [SerializeField] private bool selfShadows;

    [Header("Generation")]
    [SerializeField] private bool generateOnStart = true;

    [Header("Outline (local XY)")]
    public List<Vector3> outlinePoints = new List<Vector3>
    {
        new Vector3(-1f, -1f, 0f),
        new Vector3( 1f, -1f, 0f),
        new Vector3( 1f,  1f, 0f),
        new Vector3(-1f,  1f, 0f)
    };

    [Header("Debug")]
    [SerializeField] private bool drawBMeshDebug;

    [SerializeField, HideInInspector] private Mesh generatedMesh;
    [SerializeField, HideInInspector] private Vector3[] generatedOutline = new Vector3[0];
    [SerializeField, HideInInspector] private int generationVersion;

    private BMesh debugBMesh;

    public IReadOnlyList<Vector3> GeneratedOutline
    {
        get { return generatedOutline; }
    }

    public int GenerationVersion
    {
        get { return generationVersion; }
    }

    private void Reset()
    {
        outlinePoints = new List<Vector3>
        {
            new Vector3(-1f, -1f, 0f),
            new Vector3( 1f, -1f, 0f),
            new Vector3( 1f,  1f, 0f),
            new Vector3(-1f,  1f, 0f)
        };
    }

    private void OnEnable()
    {
#if UNITY_EDITOR
        if (!Application.isPlaying)
        {
            UnityEditor.EditorApplication.delayCall -= RegenerateAfterEnable;
            UnityEditor.EditorApplication.delayCall += RegenerateAfterEnable;
        }
#endif
    }

    private void OnDisable()
    {
#if UNITY_EDITOR
        UnityEditor.EditorApplication.delayCall -= RegenerateAfterEnable;
#endif
    }

#if UNITY_EDITOR
    private void RegenerateAfterEnable()
    {
        if (this == null || Application.isPlaying || UnityEditor.EditorUtility.IsPersistent(this))
            return;

        Regenerate();
    }
#endif

    private void Start()
    {
        if (Application.isPlaying && generateOnStart)
            Regenerate();
    }

    /// <summary>
    /// Replaces the editable outline only. It deliberately does not regenerate
    /// the mesh, physics shapes or shadow.
    /// </summary>
    public void SetOutlinePositions(IReadOnlyList<Vector3> positions)
    {
        outlinePoints.Clear();

        if (positions == null)
            return;

        for (int i = 0; i < positions.Count; i++)
            outlinePoints.Add(positions[i]);
    }

    /// <summary>
    /// Generates the mesh, PhysicsCore2D shapes and ShadowCaster2D source data
    /// from the current outlinePoints list.
    /// </summary>
    public void Regenerate()
    {
        List<Vector3> points = GetCleanOutline();

        if (!ValidateOutline(points))
            return;

        List<int> triangleIndices;

        if (!TryTriangulate(points, out triangleIndices))
        {
            Debug.LogError(
                "BMeshTerrain2D: Could not triangulate the outline. " +
                "Make sure it is a simple, non-self-intersecting polygon.",
                this
            );
            return;
        }

        GenerateMesh(points, triangleIndices);

        // Keep a separate snapshot. Changes to outlinePoints do not affect the
        // shadow until Regenerate() is called again.
        generatedOutline = points.ToArray();
        generationVersion++;

        GeneratePhysics(points);
        GenerateShadowCaster();

#if UNITY_EDITOR
        if (!Application.isPlaying)
            UnityEditor.EditorUtility.SetDirty(this);
#endif
    }

    private List<Vector3> GetCleanOutline()
    {
        List<Vector3> points = new List<Vector3>();

        if (outlinePoints == null)
            return points;

        for (int i = 0; i < outlinePoints.Count; i++)
        {
            Vector3 point = outlinePoints[i];

            if (!IsFinite(point.x) || !IsFinite(point.y) || !IsFinite(point.z))
                continue;

            if (points.Count > 0)
            {
                Vector2 previous = new Vector2(
                    points[points.Count - 1].x,
                    points[points.Count - 1].y
                );

                Vector2 current = new Vector2(point.x, point.y);

                if ((previous - current).sqrMagnitude <= 0.000001f)
                    continue;
            }

            points.Add(point);
        }

        // The polygon is implicitly closed. Remove a repeated final copy of
        // the first vertex if the caller supplied one.
        if (points.Count > 2)
        {
            Vector2 first = new Vector2(points[0].x, points[0].y);
            Vector2 last = new Vector2(
                points[points.Count - 1].x,
                points[points.Count - 1].y
            );

            if ((first - last).sqrMagnitude <= 0.000001f)
                points.RemoveAt(points.Count - 1);
        }

        // Keep the working polygon counter-clockwise in XY.
        if (CalculateSignedArea(points) < 0f)
            points.Reverse();

        return points;
    }

    private bool IsFinite(float value)
    {
        return !float.IsNaN(value) && !float.IsInfinity(value);
    }

    private float CalculateSignedArea(List<Vector3> points)
    {
        float twiceArea = 0f;

        for (int i = 0; i < points.Count; i++)
        {
            Vector3 a = points[i];
            Vector3 b = points[(i + 1) % points.Count];
            twiceArea += a.x * b.y - b.x * a.y;
        }

        return twiceArea * 0.5f;
    }

    private bool ValidateOutline(List<Vector3> points)
    {
        if (points.Count < 3)
        {
            Debug.LogError(
                "BMeshTerrain2D: An outline needs at least three distinct points.",
                this
            );
            return false;
        }

        if (Mathf.Abs(CalculateSignedArea(points)) < 0.000001f)
        {
            Debug.LogError("BMeshTerrain2D: The outline has zero area.", this);
            return false;
        }

        return true;
    }

    // ------------------------------------------------------------
    // Triangulation: ear clipping for concave simple polygons.
    // Returned indices reference the original points list.
    // ------------------------------------------------------------

    private bool TryTriangulate(List<Vector3> points, out List<int> triangles)
    {
        const float epsilon = 0.000001f;

        triangles = new List<int>();
        List<int> remaining = new List<int>(points.Count);

        for (int i = 0; i < points.Count; i++)
            remaining.Add(i);

        // Collinear points are not useful to the ear-clipping pass. They stay
        // in the original outline and remain available to physics and shadows.
        bool removedCollinear = true;

        while (removedCollinear && remaining.Count > 3)
        {
            removedCollinear = false;

            for (int i = 0; i < remaining.Count; i++)
            {
                int previousIndex = remaining[(i - 1 + remaining.Count) % remaining.Count];
                int currentIndex = remaining[i];
                int nextIndex = remaining[(i + 1) % remaining.Count];

                Vector2 previous = points[previousIndex];
                Vector2 current = points[currentIndex];
                Vector2 next = points[nextIndex];

                float cross = Cross2D(current - previous, next - current);
                float dot = Vector2.Dot(current - previous, current - next);

                if (Mathf.Abs(cross) <= epsilon && dot <= epsilon)
                {
                    remaining.RemoveAt(i);
                    removedCollinear = true;
                    break;
                }
            }
        }

        if (remaining.Count < 3)
            return false;

        int safety = 0;
        int maximumIterations = points.Count * points.Count;

        while (remaining.Count > 3 && safety < maximumIterations)
        {
            bool foundEar = false;

            for (int i = 0; i < remaining.Count; i++)
            {
                int previousIndex = remaining[(i - 1 + remaining.Count) % remaining.Count];
                int currentIndex = remaining[i];
                int nextIndex = remaining[(i + 1) % remaining.Count];

                Vector2 a = points[previousIndex];
                Vector2 b = points[currentIndex];
                Vector2 c = points[nextIndex];

                // The working polygon is counter-clockwise; a convex corner
                // therefore has a positive 2D cross product.
                if (Cross2D(b - a, c - b) <= epsilon)
                    continue;

                bool containsVertex = false;

                for (int j = 0; j < remaining.Count; j++)
                {
                    int testIndex = remaining[j];

                    if (testIndex == previousIndex ||
                        testIndex == currentIndex ||
                        testIndex == nextIndex)
                    {
                        continue;
                    }

                    Vector2 testPoint = points[testIndex];

                    if (PointInTriangle(testPoint, a, b, c))
                    {
                        containsVertex = true;
                        break;
                    }
                }

                if (containsVertex)
                    continue;

                triangles.Add(previousIndex);
                triangles.Add(currentIndex);
                triangles.Add(nextIndex);

                remaining.RemoveAt(i);
                foundEar = true;
                break;
            }

            if (!foundEar)
                return false;

            safety++;
        }

        if (remaining.Count != 3)
            return false;

        Vector2 finalA = points[remaining[0]];
        Vector2 finalB = points[remaining[1]];
        Vector2 finalC = points[remaining[2]];

        if (Cross2D(finalB - finalA, finalC - finalA) <= epsilon)
            return false;

        triangles.Add(remaining[0]);
        triangles.Add(remaining[1]);
        triangles.Add(remaining[2]);

        return true;
    }

    private float Cross2D(Vector2 a, Vector2 b)
    {
        return a.x * b.y - a.y * b.x;
    }

    private bool PointInTriangle(Vector2 point, Vector2 a, Vector2 b, Vector2 c)
    {
        const float epsilon = 0.000001f;

        float ab = Cross2D(b - a, point - a);
        float bc = Cross2D(c - b, point - b);
        float ca = Cross2D(a - c, point - c);

        return ab >= -epsilon && bc >= -epsilon && ca >= -epsilon;
    }

    // ------------------------------------------------------------
    // BMesh / Unity Mesh generation
    // ------------------------------------------------------------

    private void GenerateMesh(List<Vector3> points, List<int> triangleIndices)
    {
        BMesh bmesh = new BMesh();

        bmesh.AddVertexAttribute("color", AttributeBaseType.Float, 3);
        bmesh.AddVertexAttribute("uv", AttributeBaseType.Float, 2);

        float minX = points[0].x;
        float maxX = points[0].x;
        float minY = points[0].y;
        float maxY = points[0].y;

        for (int i = 1; i < points.Count; i++)
        {
            minX = Mathf.Min(minX, points[i].x);
            maxX = Mathf.Max(maxX, points[i].x);
            minY = Mathf.Min(minY, points[i].y);
            maxY = Mathf.Max(maxY, points[i].y);
        }

        float width = Mathf.Max(maxX - minX, 0.000001f);
        float height = Mathf.Max(maxY - minY, 0.000001f);

        Vertex[] vertices = new Vertex[points.Count];

        for (int i = 0; i < points.Count; i++)
        {
            Vector3 point = points[i];
            vertices[i] = bmesh.AddVertex(point.x, point.y, point.z);

            // This BMeshUnity package exposes FloatAttributeValue.data as a
            // public float array. Initialize this Float3 attribute through the array.
            FloatAttributeValue colorAttribute = new FloatAttributeValue();
            colorAttribute.data = new float[]
            {
                vertexColor.r,
                vertexColor.g,
                vertexColor.b
            };

            vertices[i].attributes["color"] = colorAttribute;

            vertices[i].attributes["uv"] = new FloatAttributeValue(
                (point.x - minX) / width,
                (point.y - minY) / height
            );
        }

        // BMeshUnity's Unity mesh conversion accepts triangle and quad faces.
        // Build triangles here instead of creating one unsupported n-gon face.
        for (int i = 0; i < triangleIndices.Count; i += 3)
        {
            bmesh.AddFace(
                vertices[triangleIndices[i]],
                vertices[triangleIndices[i + 1]],
                vertices[triangleIndices[i + 2]]
            );
        }

        MeshFilter meshFilter = GetComponent<MeshFilter>();
        MeshRenderer meshRenderer = GetComponent<MeshRenderer>();
        Mesh previousGeneratedMesh = generatedMesh;

        BMeshUnity.SetInMeshFilter(bmesh, meshFilter);

        generatedMesh = meshFilter.sharedMesh;
        meshRenderer.sharedMaterial = material;
        debugBMesh = bmesh;

        if (previousGeneratedMesh != null && previousGeneratedMesh != generatedMesh)
            DestroyGeneratedObject(previousGeneratedMesh);
    }

    // ------------------------------------------------------------
    // PhysicsCore2D generation
    // ------------------------------------------------------------

    private void GeneratePhysics(List<Vector3> points)
    {
        PhysicsPose physicsPose = GetComponent<PhysicsPose>();
        PhysicsBody body = physicsPose.body;

        if (!body.isValid)
        {
            Debug.LogWarning(
                "BMeshTerrain2D: PhysicsPose does not currently have a valid " +
                "PhysicsCore2D body. Mesh and shadow were generated, but physics was skipped.",
                this
            );
            return;
        }

        Vector2[] physicsPoints = new Vector2[points.Count];

        for (int i = 0; i < points.Count; i++)
            physicsPoints[i] = new Vector2(points[i].x, points[i].y);

        using (NativeArray<PolygonGeometry> polygons = PolygonGeometry.CreatePolygons(
            physicsPoints,
            PhysicsTransform.identity,
            Vector2.one,
            Allocator.Temp
        ))
        {
            if (polygons.Length == 0)
            {
                Debug.LogError(
                    "BMeshTerrain2D: PhysicsCore2D returned no polygon geometry.",
                    this
                );
                return;
            }

            // Apply the selected PhysicsCore2D layer to the generated shapes.
            // Only categories are changed; contacts and groupIndex are preserved.
            PhysicsShapeDefinition shapeDefinition = physicsShapeDefinition;

            if (!string.IsNullOrWhiteSpace(physicsLayerName))
            {
                PhysicsShape.ContactFilter contactFilter = shapeDefinition.contactFilter;
                contactFilter.categories = PhysicsLayers.GetLayerMask(physicsLayerName);
                shapeDefinition.contactFilter = contactFilter;
            }

            // This script assumes it owns all shapes on this PhysicsPose body.
            // Remove the previous generated set before creating the replacement.
            ClearExistingPhysicsShapes(body);

            using (NativeArray<PhysicsShape> createdShapes = body.CreateShapeBatch(
                polygons,
                shapeDefinition,
                Allocator.Temp
            ))
            {
                if (createdShapes.Length == 0)
                {
                    Debug.LogError(
                        "BMeshTerrain2D: PhysicsCore2D failed to create physics shapes.",
                        this
                    );
                    return;
                }
            }

            body.ApplyMassFromShapes();
        }
    }

    private void ClearExistingPhysicsShapes(PhysicsBody body)
    {
        using (NativeArray<PhysicsShape> existingShapes = body.GetShapes(Allocator.Temp))
        {
            for (int i = 0; i < existingShapes.Length; i++)
                existingShapes[i].Destroy(false, 0);
        }

        body.ApplyMassFromShapes();
    }

    // ------------------------------------------------------------
    // ShadowCaster2D setup
    // ------------------------------------------------------------

    private void GenerateShadowCaster()
    {
        if (!generateShadowCaster)
            return;

        ShadowCaster2D caster = GetComponent<ShadowCaster2D>();

        if (caster == null)
            caster = gameObject.AddComponent<ShadowCaster2D>();

        // Geometry itself is supplied by BMeshTerrainShadowShapeProvider,
        // selected once in the ShadowCaster2D Casting Source dropdown.
    }

    // ------------------------------------------------------------
    // Debug / cleanup
    // ------------------------------------------------------------

    private void OnDrawGizmos()
    {
        if (!drawBMeshDebug || debugBMesh == null)
            return;

        Matrix4x4 previousMatrix = Gizmos.matrix;
        Gizmos.matrix = transform.localToWorldMatrix;
        BMeshUnity.DrawGizmos(debugBMesh);
        Gizmos.matrix = previousMatrix;
    }

    private void DestroyGeneratedObject(Object target)
    {
        if (target == null)
            return;

        if (Application.isPlaying)
            Destroy(target);
        else
            DestroyImmediate(target);
    }
}
