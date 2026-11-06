using System.Collections;
using System.Collections.Generic;
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

    private readonly List<PhysicsBody> _bodiesOrdered = new List<PhysicsBody>();
    private readonly Dictionary<int, PhysicsBody> _bodiesByIndex = new Dictionary<int, PhysicsBody>();

    public IReadOnlyList<PhysicsBody> Bodies => _bodiesOrdered;
    public IReadOnlyDictionary<int, PhysicsBody> BodiesByIndex => _bodiesByIndex;
    public int SpawnedCount => _bodiesOrdered.Count;
    public bool IsSpawning { get; private set; }

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
    }

    public bool TryGetBody(int index, out PhysicsBody data)
    {
        return _bodiesByIndex.TryGetValue(index, out data);
    }

    public PhysicsBody GetBody(int index)
    {
        return _bodiesByIndex[index];
    }

    public void ClearRecords()
    {
        _bodiesOrdered.Clear();
        _bodiesByIndex.Clear();
    }

    void Spawn()
    {
        if (IsSpawning) return;
        ClearRecords();
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
                    
                }
            );
            Debug.Log($"Body {i} valid: {body.isValid} pos: {worldPos}");
            PhysicsShapeDefinition shapeDef = new PhysicsShapeDefinition
                {
                    triggerEvents = true,

                    surfaceMaterial = new PhysicsShape.SurfaceMaterial
                    {
                        friction = 0.5f,
                        customColor = Color.red,
                    }
                };
            PhysicsShape createdShape = body.CreateShape(
                new CircleGeometry { center = Vector2.zero, radius = 0.18f },
                shapeDef
            );
            _bodiesOrdered.Add(body);
            _bodiesByIndex[i] = body;
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
}
