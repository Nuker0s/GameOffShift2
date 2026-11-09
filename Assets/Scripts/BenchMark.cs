using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using UnityEngine;
using Unity.U2D.Physics;
using Unity.U2D.Physics.Assets;

public class BenchMark : MonoBehaviour
{

    public bool spawn = false;
    public int spawnCount = 100;
    [Tooltip("How many bodies to spawn per frame-batch before yielding.")]
    public int spawnPerFrame = 10;
    [Tooltip("Spacing between bodies in world units.")]
    public float spacing = 0.3f;
    [Tooltip("Extra delay between batches. 0 = next frame.")]
    public float batchDelay = 0f;
    bool IsSpawning = false;

    [Header("Particle Properties")]
    [Tooltip("Radius of the spawned circle shape. Applies to newly spawned bodies. Changing radius requires respawn (geometry).")]
    public float particleRadius = 0.10f;
    [Tooltip("Density/weight of the shape. Also applied to already-spawned bodies when Apply is used.")]
    public float particleDensity = 0.2f;
    [Tooltip("Friction of the shape surface material.")]
    public float particleFriction = 0.2f;
    [Tooltip("Bounciness of the shape surface material.")]
    public float particleBounciness = 0.2f;
    [Tooltip("Color of the shape surface material (customColor).")]
    public Color particleColor = Color.red;
    [Tooltip("Mass of the body (massConfiguration.mass).")]
    public float particleMass = 1f;
    [Tooltip("Linear damping of the body.")]
    public float particleLinearDamping = 0.2f;
    [Tooltip("Gravity scale of the body.")]
    public float particleGravityScale = 1f;
    [Tooltip("If true, shape trigger events are enabled.")]
    public bool particleTriggerEvents = true;
    [Tooltip("If true, rotation is frozen (constraints = Rotation).")]
    public bool particleFreezeRotation = true;
    [Tooltip("Click (set true) to push the above properties to already-spawned bodies/shapes.")]
    public bool applyParticlePropertiesNow = false;
    [Tooltip("If true, any inspector change to the properties above auto-applies to already-spawned bodies every frame / on validate.")]
    public bool liveUpdateExistingParticles = false;

    [Header("Particle Types")]
    [Tooltip("Type id assigned to newly spawned bodies. Stored in Vector3.z sent to the shader.")]
    public float defaultParticleType = 0f;
    [Tooltip("Click (set true) to push Default Particle Type to ALL already-spawned bodies.")]
    public bool applyParticleTypeToExisting = false;

    [Header("Metaball Rendering")]
    [Tooltip("Renderer of the plane/quad that draws the metaballs (uses world-space XY).")]
    public Renderer metaballPlaneRenderer;
    [Tooltip("Optional. If set, this material is updated. Otherwise the renderer's sharedMaterial is used.")]
    public Material metaballMaterialOverride;
    [Tooltip("Starting ComputeBuffer capacity. Grows automatically as needed.")]
    public int metaballBufferInitialCapacity = 512;
    [Tooltip("Metaball field radius. Must roughly match _Radius in the shader.")]
    public float metaballRadius = 0.4f;
    [Tooltip("Field threshold for surface. Must roughly match _Threshold in the shader.")]
    public float metaballThreshold = 1.0f;
    [Tooltip("If true, material float props are synced from here every frame.")]
    public bool syncMaterialParams = true;

    [Header("Debug")]
    public bool showDebugLogs = true;
    [Tooltip("If true and no bodies exist yet, renders one fake metaball at world origin so you can verify the quad/shader/camera without spawning physics.")]
    public bool debugTestBallAtOrigin = true;
    public float debugTestBallType = 0f;

    private readonly List<PhysicsBody> _bodiesOrdered = new List<PhysicsBody>();
    private readonly List<float> _bodyTypesOrdered = new List<float>();
    private readonly List<PhysicsShape> _shapesOrdered = new List<PhysicsShape>();

    // CPU mirror of GPU data: xy = world XY, z = type id.
    private Vector3[] _metaballData;
    private ComputeBuffer _metaballBuffer;

    private static readonly int MetaballsId = Shader.PropertyToID("_Metaballs");
    private static readonly int MetaballCountId = Shader.PropertyToID("_MetaballCount");
    private static readonly int RadiusId = Shader.PropertyToID("_Radius");
    private static readonly int ThresholdId = Shader.PropertyToID("_Threshold");

    private PhysicsWorld physicsWorld;

    void Start()
    {
        physicsWorld = PhysicsWorld.defaultWorld;
        physicsWorld.drawOptions = PhysicsWorld.DrawOptions.AllShapes;
    }

    void Update()
    {
        if (spawn)
        {
            spawn = false;
            Spawn();
        }

        if (applyParticlePropertiesNow)
        {
            applyParticlePropertiesNow = false;
            ApplyParticlePropertiesToSpawnedBodies();
        }

        if (applyParticleTypeToExisting)
        {
            applyParticleTypeToExisting = false;
            ApplyParticleTypeToAllBodies();
        }

        if (liveUpdateExistingParticles)
        {
            ApplyParticlePropertiesToSpawnedBodies();
        }
    }

    void OnValidate()
    {
        if (liveUpdateExistingParticles && Application.isPlaying)
        {
            ApplyParticlePropertiesToSpawnedBodies();
        }
    }

    void LateUpdate()
    {
        PushMetaballPositionsToShader();
    }

    void OnDestroy()
    {
        ReleaseMetaballBuffer();
    }

    /// <summary>
    /// Change the visual type of a registered body (goes into Vector3.z for the shader).
    /// </summary>
    public void SetBodyType(int index, float typeId)
    {
        if (index < 0 || index >= _bodyTypesOrdered.Count) return;
        _bodyTypesOrdered[index] = typeId;
    }

    /// <summary>Read-only access to bodies for type bookkeeping / debugging.</summary>
    public IReadOnlyList<PhysicsBody> Bodies => _bodiesOrdered;

    /// <summary>Read-only access to the shapes created for each spawned body (parallel to Bodies).</summary>
    public IReadOnlyList<PhysicsShape> Shapes => _shapesOrdered;

    public void ClearRecords()
    {
        _bodiesOrdered.Clear();
        _bodyTypesOrdered.Clear();
        _shapesOrdered.Clear();
    }

    /// <summary>
    /// Applies the exposed particle properties to every already-spawned body/shape.
    /// NOTE: radius is geometry — it only affects newly spawned bodies.
    /// Uses reflection for body/shape setters so unknown API names fail softly (warn) instead of breaking compile.
    /// </summary>
    public void ApplyParticlePropertiesToSpawnedBodies()
    {
        for (int i = 0; i < _bodiesOrdered.Count; i++)
        {
            bool hasShape = i < _shapesOrdered.Count;
            ApplyParticlePropertiesToIndexedBody(i, hasShape);
        }
    }

    /// <summary>
    /// Sets every already-spawned body to <see cref="defaultParticleType"/>.
    /// </summary>
    public void ApplyParticleTypeToAllBodies()
    {
        for (int i = 0; i < _bodyTypesOrdered.Count; i++)
            _bodyTypesOrdered[i] = defaultParticleType;
    }

    void ApplyParticlePropertiesToIndexedBody(int index, bool hasShape)
    {
        PhysicsBody body = _bodiesOrdered[index];
        if (body == null) return;

        TrySetMember(body, "linearDamping", particleLinearDamping);
        TrySetMember(body, "gravityScale", particleGravityScale);
        TrySetMember(body, "mass", particleMass);

        // constraints is an enum on PhysicsBody; set via Enum parse-safe path.
        object rotationFlag = GetEnumValue(typeof(PhysicsBody.BodyConstraints), "Rotation");
        object noneFlag = GetEnumValue(typeof(PhysicsBody.BodyConstraints), "None");
        if (rotationFlag != null && noneFlag != null)
            TrySetMember(body, "constraints", particleFreezeRotation ? rotationFlag : noneFlag);

        if (!hasShape) return;

        // PhysicsShape is a struct handle — must box, mutate, then write back.
        object shapeBoxed = _shapesOrdered[index];
        TrySetMember(shapeBoxed, "density", particleDensity);
        TrySetMember(shapeBoxed, "triggerEvents", particleTriggerEvents);

        // surfaceMaterial is a nested struct: read, mutate, write back.
        object surfMat = GetMemberValue(shapeBoxed, "surfaceMaterial");
        if (surfMat != null)
        {
            TrySetMember(surfMat, "friction", particleFriction);
            TrySetMember(surfMat, "bounciness", particleBounciness);
            TrySetMember(surfMat, "customColor", particleColor);
            TrySetMember(shapeBoxed, "surfaceMaterial", surfMat);
        }

        _shapesOrdered[index] = (PhysicsShape)shapeBoxed;
    }

    static void TrySetMember(object target, string memberName, object value)
    {
        if (target == null || string.IsNullOrEmpty(memberName)) return;
        System.Type t = target.GetType();

        // Property first (covers most Unity setters).
        PropertyInfo prop = t.GetProperty(memberName,
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
        if (prop != null && prop.CanWrite)
        {
            try
            {
                object converted = ConvertValue(value, prop.PropertyType);
                prop.SetValue(target, converted);
                return;
            }
            catch { /* fall through to field, then warn */ }
        }

        FieldInfo field = t.GetField(memberName,
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
        if (field != null)
        {
            try
            {
                object converted = ConvertValue(value, field.FieldType);
                field.SetValue(target, converted);
                return;
            }
            catch { /* warn below */ }
        }

        Debug.LogWarning($"[BenchMark] No writable property/field '{memberName}' on {t.Name}; skipped.");
    }

    static object GetMemberValue(object target, string memberName)
    {
        if (target == null) return null;
        System.Type t = target.GetType();
        PropertyInfo prop = t.GetProperty(memberName,
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
        if (prop != null && prop.CanRead)
        {
            try { return prop.GetValue(target); } catch { return null; }
        }
        FieldInfo field = t.GetField(memberName,
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
        if (field != null)
        {
            try { return field.GetValue(target); } catch { return null; }
        }
        Debug.LogWarning($"[BenchMark] No readable property/field '{memberName}' on {t.Name}; skipped.");
        return null;
    }

    static object GetEnumValue(System.Type enumType, string name)
    {
        try
        {
            if (enumType != null && enumType.IsEnum)
                return System.Enum.Parse(enumType, name);
        }
        catch { }
        return null;
    }

    static object ConvertValue(object value, System.Type targetType)
    {
        if (value == null) return null;
        if (targetType.IsInstanceOfType(value)) return value;
        if (targetType.IsEnum && value is string s) return System.Enum.Parse(targetType, s);
        return System.Convert.ChangeType(value, targetType);
    }

    void Spawn()
    {
        if (IsSpawning) return;
        //ClearRecords();
        StartCoroutine(SpawnOverTime());
    }

    IEnumerator SpawnOverTime()
    {
        IsSpawning = true;
        int columns = Mathf.CeilToInt(Mathf.Sqrt(spawnCount));
        columns = Mathf.Max(1, columns);
        int rows = Mathf.CeilToInt((float)spawnCount / columns);
        rows = Mathf.Max(1, rows);
        Vector2 gridSize = new Vector2((columns - 1) * spacing, (rows - 1) * spacing);
        Vector2 centerOffset = gridSize * 0.5f;
        int batchCounter = 0;
        int perBatch = Mathf.Max(1, spawnPerFrame);
        for (int i = 0; i < spawnCount; i++)
        {
            int col = i % columns;
            int row = i / columns;
            Vector2 gridPos = new Vector2(col * spacing, row * spacing);
            Vector2 worldPos = gridPos - centerOffset;
            PhysicsBody body = physicsWorld.CreateBody(
                new PhysicsBodyDefinition
                {
                    type = PhysicsBody.BodyType.Dynamic,
                    position = worldPos,
                    rotation = PhysicsRotate.down,
                    linearDamping = particleLinearDamping,
                    gravityScale = particleGravityScale,
                    constraints = particleFreezeRotation
                        ? PhysicsBody.BodyConstraints.Rotation
                        : PhysicsBody.BodyConstraints.None,
                    massConfiguration = new PhysicsBody.MassConfiguration
                    {
                        mass = particleMass,

                    }
                    
                }
            );
            PhysicsShapeDefinition shapeDef = new PhysicsShapeDefinition
                {
                    triggerEvents = particleTriggerEvents,
                    density = particleDensity, //weight

                    surfaceMaterial = new PhysicsShape.SurfaceMaterial
                    {
                        customColor = particleColor,
                        friction = particleFriction,
                        bounciness = particleBounciness,
                    }
                };
            PhysicsShape createdShape = body.CreateShape(
                new CircleGeometry { center = Vector2.zero, radius = particleRadius },
                shapeDef
            );
            _bodiesOrdered.Add(body);
            _shapesOrdered.Add(createdShape);
            _bodyTypesOrdered.Add(defaultParticleType);
            batchCounter++;
            if (batchCounter >= perBatch && i < spawnCount - 1)
            {
                batchCounter = 0;
                if (batchDelay > 0f)
                    yield return new WaitForSeconds(batchDelay);
                else
                    yield return null;
            }
        }
        IsSpawning = false;
    }

    Material ResolveMetaballMaterial()
    {
        if (metaballMaterialOverride != null)
            return metaballMaterialOverride;
        if (metaballPlaneRenderer != null)
        {
            // Use .material (instance) instead of .sharedMaterial so this
            // component's SetBuffer/SetInt calls don't leak into other
            // objects and aren't stomped by SRP batcher state.
            return metaballPlaneRenderer.material;
        }
        return null;
    }

    void EnsureMetaballCapacity(int needed)
    {
        if (needed <= 0) return;
        if (_metaballBuffer != null && _metaballBuffer.count >= needed) return;

        int newCap = Mathf.Max(Mathf.Max(1, metaballBufferInitialCapacity), needed);
        if (_metaballBuffer != null)
        {
            // Grow: double until it fits.
            newCap = Mathf.Max(newCap, _metaballBuffer.count * 2);
            ReleaseMetaballBuffer();
        }

        // float3 = 3 floats = 12 bytes.
        _metaballBuffer = new ComputeBuffer(newCap, sizeof(float) * 3);
        _metaballData = new Vector3[newCap];
    }

    void ReleaseMetaballBuffer()
    {
        if (_metaballBuffer != null)
        {
            _metaballBuffer.Release();
            _metaballBuffer = null;
        }
        _metaballData = null;
    }

    /// <summary>
    /// Copies xy = body world position, z = particle type into the shader's StructuredBuffer.
    /// Call every frame (LateUpdate) so the metaball plane follows physics.
    /// </summary>
    void PushMetaballPositionsToShader()
    {
        Material mat = ResolveMetaballMaterial();
        if (mat == null)
        {
            if (showDebugLogs) Debug.LogWarning("[Metaballs] No material: assign Metaball Plane Renderer or Material Override.");
            return;
        }

        int count = _bodiesOrdered.Count;

        // Debug fallback: if nothing spawned yet, draw one test ball at
        // origin so an empty quad / bad camera setup is distinguishable
        // from "no data pushed".
        bool usingTestBall = false;
        if (count == 0 && debugTestBallAtOrigin)
        {
            usingTestBall = true;
            count = 1;
        }

        mat.SetInt(MetaballCountId, count);
        if (count == 0) return;

        EnsureMetaballCapacity(count);
        if (_metaballBuffer == null || _metaballData == null) return;

        if (usingTestBall)
        {
            _metaballData[0] = new Vector3(0f, 0f, debugTestBallType);
        }
        else
        {
            for (int i = 0; i < count; i++)
            {
                Vector2 p = _bodiesOrdered[i].position;
                _metaballData[i].x = p.x;
                _metaballData[i].y = p.y;
                _metaballData[i].z = _bodyTypesOrdered[i];
            }
        }

        _metaballBuffer.SetData(_metaballData, 0, 0, count);
        mat.SetBuffer(MetaballsId, _metaballBuffer);

        if (syncMaterialParams)
        {
            mat.SetFloat(RadiusId, metaballRadius);
            mat.SetFloat(ThresholdId, metaballThreshold);
        }

        if (showDebugLogs && Time.frameCount % 60 == 0)
        {
            string first = usingTestBall ? "(test ball at origin)"
                : _metaballData[0].ToString();
            Debug.Log($"[Metaballs] count={count} first={first} " +
                      $"radius={metaballRadius} threshold={metaballThreshold} " +
                      $"mat={mat.name} buf={_metaballBuffer.count}");
        }
    }
}
