
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using Unity.Collections;
using Unity.U2D.Physics;

public class RoomTransporter : MonoBehaviour
{
    public string ID;
    public string toLoad;
    public string transporterTarget;

    public static string PersistentSceneName = "";

    public int doorToOpen;

    public Vector2 MoveTarget;
    public GameObject[] doors;

    [Header("Transported Physics Bodies")]
    public Vector2 transportBoxSize = new Vector2(5f, 5f);
    public Vector2 transportBoxOffset = Vector2.zero;

    public bool transportDebug = false;

    private struct TransportedBody
    {
        public PhysicsBody body;
        public Vector2 relativePosition;
        public float relativeRotation;
        public GameObject rootObject;
    }

    public void Tranport()
    {
        CloseDoor(0);
        CloseDoor(1);
        CloseDoor(2);
        CloseDoor(3);

        StartCoroutine(TransportRoutine());
    }

    public void Transport()
    {
        Tranport();
    }

    private IEnumerator TransportRoutine()
    {
        // Capture every eligible body before unloading the current scene.
        List<TransportedBody> transportedBodies = CaptureTransportedBodies();

        // Capture current scene before moving to DDOL.
        Scene currentScene = gameObject.scene;

        // Move transporter into Don't Destroy On Load.
        DontDestroyOnLoad(gameObject);

        // Unload current scene, except for the persistent scene.
        bool isCurrentPersistent = currentScene.IsValid()
            && !string.IsNullOrEmpty(PersistentSceneName)
            && currentScene.name == PersistentSceneName;

        if (currentScene.IsValid() && !isCurrentPersistent)
        {
            AsyncOperation unload = SceneManager.UnloadSceneAsync(currentScene);

            while (unload != null && !unload.isDone)
            {
                yield return null;
            }
        }

        // Teleport transporter to its intermediate target.
        transform.position = new Vector3(
            MoveTarget.x,
            MoveTarget.y,
            transform.position.z
        );

        // Load destination scene additively.
        AsyncOperation load = SceneManager.LoadSceneAsync(
            toLoad,
            LoadSceneMode.Additive
        );

        while (load != null && !load.isDone)
        {
            yield return null;
        }

        // Allow new scene objects to initialise.
        yield return null;

        Scene newScene = SceneManager.GetSceneByName(toLoad);

        if (newScene.IsValid() && newScene.isLoaded)
        {
            SceneManager.SetActiveScene(newScene);
        }

        // Find destination transporter.
        RoomTransporter target = null;

        RoomTransporter[] all = FindObjectsByType<RoomTransporter>(
            FindObjectsInactive.Exclude
        );

        for (int i = 0; i < all.Length; i++)
        {
            if (all[i] != this && all[i].ID == transporterTarget)
            {
                target = all[i];
                break;
            }
        }

        if (target == null)
        {
            // No target found: move everything into the loaded fallback scene.
            Scene fallback = SceneManager.GetSceneByName(toLoad);

            if (fallback.IsValid() && fallback.isLoaded)
            {
                SceneManager.MoveGameObjectToScene(gameObject, fallback);

                MoveTransportedBodies(transportedBodies, fallback);
            }

            yield break;
        }

        // Store destination information before destroying the target.
        Scene destinationScene = target.gameObject.scene;

        Vector3 destinationPosition = target.transform.position;
        Quaternion destinationRotation = target.transform.rotation;

        // Move transporter into the destination scene.
        SceneManager.MoveGameObjectToScene(gameObject, destinationScene);

        // Teleport and rotate transporter.
        transform.SetPositionAndRotation(
            destinationPosition,
            destinationRotation
        );

        // Move all captured Physics Core 2D bodies with it.
        MoveTransportedBodies(transportedBodies, destinationScene);

        // Copy destination transporter settings.
        ID = target.ID;
        toLoad = target.toLoad;
        transporterTarget = target.transporterTarget;
        doorToOpen = target.doorToOpen;
        MoveTarget = target.MoveTarget;

        // Destroy destination transporter.
        Destroy(target.gameObject);
        OpenDoor(doorToOpen);
    }

    private List<TransportedBody> CaptureTransportedBodies()
    {
        List<TransportedBody> capturedBodies = new List<TransportedBody>();

        Vector2 safeSize = new Vector2(
            Mathf.Abs(transportBoxSize.x),
            Mathf.Abs(transportBoxSize.y)
        );

        Vector2 halfSize = safeSize * 0.5f;

        // Calculate the four world-space corners of our oriented box.
        Vector3 corner0 = transform.TransformPoint(
            new Vector3(
                transportBoxOffset.x - halfSize.x,
                transportBoxOffset.y - halfSize.y,
                0f
            )
        );

        Vector3 corner1 = transform.TransformPoint(
            new Vector3(
                transportBoxOffset.x + halfSize.x,
                transportBoxOffset.y - halfSize.y,
                0f
            )
        );

        Vector3 corner2 = transform.TransformPoint(
            new Vector3(
                transportBoxOffset.x + halfSize.x,
                transportBoxOffset.y + halfSize.y,
                0f
            )
        );

        Vector3 corner3 = transform.TransformPoint(
            new Vector3(
                transportBoxOffset.x - halfSize.x,
                transportBoxOffset.y + halfSize.y,
                0f
            )
        );

        // Calculate the axis-aligned bounding box containing our rotated box.
        float minX = Mathf.Min(
            Mathf.Min(corner0.x, corner1.x),
            Mathf.Min(corner2.x, corner3.x)
        );

        float minY = Mathf.Min(
            Mathf.Min(corner0.y, corner1.y),
            Mathf.Min(corner2.y, corner3.y)
        );

        float maxX = Mathf.Max(
            Mathf.Max(corner0.x, corner1.x),
            Mathf.Max(corner2.x, corner3.x)
        );

        float maxY = Mathf.Max(
            Mathf.Max(corner0.y, corner1.y),
            Mathf.Max(corner2.y, corner3.y)
        );

        PhysicsAABB queryAABB = new PhysicsAABB(
            new Vector2(minX, minY),
            new Vector2(maxX, maxY)
        );

        PhysicsWorld world = PhysicsWorld.defaultWorld;

        NativeArray<PhysicsQuery.WorldOverlapResult> results =
            world.OverlapAABB(
                queryAABB,
                PhysicsQuery.QueryFilter.Everything,
                Allocator.Temp
            );

        HashSet<PhysicsBody> uniqueBodies = new HashSet<PhysicsBody>();
        HashSet<GameObject> persistentRoots = new HashSet<GameObject>();

        try
        {
            for (int i = 0; i < results.Length; i++)
            {
                PhysicsQuery.WorldOverlapResult result = results[i];

                if (!result.isValid)
                    continue;

                PhysicsShape shape = result.shape;

                if (!shape.isValid)
                    continue;

                PhysicsBody body = shape.body;

                if (!body.isValid)
                    continue;

                // Only transport dynamic and kinematic bodies.
                if (body.type != PhysicsBody.BodyType.Dynamic &&
                    body.type != PhysicsBody.BodyType.Kinematic)
                {
                    continue;
                }

                // Each body can have multiple shapes.
                if (uniqueBodies.Contains(body))
                    continue;

                // Check the body's origin against the actual rotated box.
                Vector3 localPosition = transform.InverseTransformPoint(
                    new Vector3(
                        body.position.x,
                        body.position.y,
                        transform.position.z
                    )
                );

                if (Mathf.Abs(localPosition.x - transportBoxOffset.x) > halfSize.x ||
                    Mathf.Abs(localPosition.y - transportBoxOffset.y) > halfSize.y)
                {
                    continue;
                }

                Transform bodyTransform = body.transformObject;

                GameObject rootObject = bodyTransform != null
                    ? bodyTransform.root.gameObject
                    : null;

                // Don't capture the transporter itself.
                if (rootObject == transform.root.gameObject)
                    continue;

                uniqueBodies.Add(body);

                // Store its position relative to the transporter.
                TransportedBody transportedBody = new TransportedBody
                {
                    body = body,

                    relativePosition = new Vector2(
                        localPosition.x,
                        localPosition.y
                    ),

                    relativeRotation = Mathf.DeltaAngle(
                        transform.eulerAngles.z,
                        body.rotation.degrees
                    ),

                    rootObject = rootObject
                };

                capturedBodies.Add(transportedBody);

                // Preserve the GameObject through the scene unload.
                if (rootObject != null && persistentRoots.Add(rootObject))
                {
                    DontDestroyOnLoad(rootObject);
                }
            }
        }
        finally
        {
            if (results.IsCreated)
            {
                results.Dispose();
            }
        }

        return capturedBodies;
    }

    private void MoveTransportedBodies(
        List<TransportedBody> transportedBodies,
        Scene destinationScene
    )
    {
        HashSet<GameObject> movedRoots = new HashSet<GameObject>();

        for (int i = 0; i < transportedBodies.Count; i++)
        {
            TransportedBody transportedBody = transportedBodies[i];

            PhysicsBody body = transportedBody.body;

            if (!body.isValid)
                continue;

            // Move the associated GameObject into the destination scene.
            GameObject rootObject = transportedBody.rootObject;

            if (rootObject != null &&
                rootObject != transform.root.gameObject &&
                movedRoots.Add(rootObject))
            {
                if (rootObject.scene != destinationScene)
                {
                    SceneManager.MoveGameObjectToScene(
                        rootObject,
                        destinationScene
                    );
                }
            }

            // Transform the original local position into the new world position.
            Vector3 worldPosition = transform.TransformPoint(
                new Vector3(
                    transportedBody.relativePosition.x,
                    transportedBody.relativePosition.y,
                    0f
                )
            );

            body.position = new Vector2(
                worldPosition.x,
                worldPosition.y
            );

            // Preserve its original rotation relative to the transporter.
            body.rotation = PhysicsRotate.FromDegrees(
                transform.eulerAngles.z + transportedBody.relativeRotation
            );

            // Wake it so the new pose is processed.
            body.awake = true;

            // Immediately update its associated Unity Transform.
            body.WritePose();
        }
    }

    private Vector2 GetSafeBoxSize()
    {
        return new Vector2(
            Mathf.Abs(transportBoxSize.x),
            Mathf.Abs(transportBoxSize.y)
        );
    }

    private void OnDrawGizmos()
    {
        Matrix4x4 previousMatrix = Gizmos.matrix;
        Color previousColor = Gizmos.color;

        Gizmos.matrix = transform.localToWorldMatrix;
        Gizmos.color = Color.cyan;

        Gizmos.DrawWireCube(
            transportBoxOffset,
            GetSafeBoxSize()
        );

        Gizmos.matrix = previousMatrix;
        Gizmos.color = previousColor;
    }

    void Update()
    {
        if (transportDebug)
        {
            transportDebug = false;
            Tranport();
        }
    }

    public void CloseDoor(int door)
    {
        doors[door].SetActive(true);
    }

    public void OpenDoor(int door)
    {
        doors[door].SetActive(false);
    }
}
