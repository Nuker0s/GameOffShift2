using System.Linq;
using Unity.U2D.Physics;
using UnityEngine;
using UnityEngine.InputSystem;

public class Finger : MonoBehaviour
{
    [Header("Camera To Control")]
    [Tooltip("Camera moved by edge panning. If empty, uses Camera on this GameObject, otherwise Camera.main.")]
    public Camera controlledCamera;

    [Header("Edge Pan Settings")]
    [Tooltip("World units per second the camera moves when mouse is at the edge.")]
    public float panSpeed = 20f;
    [Tooltip("Distance in pixels from the window edge that starts panning.")]
    public float borderThickness = 20f;

    [Header("Axes")]
    public bool allowHorizontal = true;
    public bool allowVertical = true;

    [Header("Bounds (optional)")]
    [Tooltip("If true, camera position is clamped after panning.")]
    public bool clampCamera = false;
    public Vector3 minBounds = new Vector3(-50f, -50f, -50f);
    public Vector3 maxBounds = new Vector3(50f, 50f, 50f);

    Transform cameraTransform;
    Camera cachedCamera;

    void Awake()
    {
        CacheCamera();
    }

    void Start()
    {
        CacheCamera();
    }

    void CacheCamera()
    {
        cachedCamera = controlledCamera;
        if (cachedCamera == null)
            cachedCamera = GetComponent<Camera>();
        if (cachedCamera == null)
            cachedCamera = Camera.main;

        if (cachedCamera != null)
            cameraTransform = cachedCamera.transform;
        else
            cameraTransform = transform;
    }

    public PhysicsPose sphere;
    public float fingerMoveMultiplier = 10f;
    public float forceMultiplier = 10f;
    void FixedUpdate()
    {
        Vector3 mousePos = Camera.main.ScreenToWorldPoint( Mouse.current.position.ReadValue());
        Vector2 worldPos = (Vector2)mousePos;
        PhysicsBody body = sphere.body;
        body.linearVelocity = (worldPos - sphere.body.position);
        //sphere.body.ApplyForce((worldPos-sphere.body.position).normalized * fingerMoveMultiplier,sphere.body.position);
        if(Mouse.current.leftButton.isPressed)
        {
            Debug.Log(sphere.body.GetShapes()[0].GetTriggerVisitors().Count());
            foreach (PhysicsShape item in sphere.body.GetShapes()[0].GetTriggerVisitors())
            {
                item.body.ApplyForce((sphere.body.position - item.body.position).normalized * Vector2.Distance(sphere.body.position, item.body.position) * forceMultiplier, item.body.position);
            }
            
        }
    }
    
    // Update is called once per frame
    void Update()
    {
        if (cameraTransform == null)
        {
            CacheCamera();
            if (cameraTransform == null)
                return;
        }

        // Re-resolve if user assigned a new camera in Inspector at runtime
        if (controlledCamera != null && controlledCamera != cachedCamera)
        {
            cachedCamera = controlledCamera;
            cameraTransform = cachedCamera.transform;
        }

        Vector3 mousePos = Mouse.current.position.ReadValue();

        // Don't pan when mouse is outside the game window (e.g. alt-tabbed).
        // Input.mousePosition can be negative or larger than Screen size in that case.
        if (mousePos.x < 0 || mousePos.x > Screen.width ||
            mousePos.y < 0 || mousePos.y > Screen.height)
            return;

        float moveX = 0f;
        float moveY = 0f; // screen up/down

        if (allowHorizontal)
        {
            if (mousePos.x >= Screen.width - borderThickness)
                moveX = 1f;
            else if (mousePos.x <= borderThickness)
                moveX = -1f;
        }

        if (allowVertical)
        {
            if (mousePos.y >= Screen.height - borderThickness)
                moveY = 1f;
            else if (mousePos.y <= borderThickness)
                moveY = -1f;
        }

        if (moveX == 0f && moveY == 0f)
            return;

        float step = panSpeed * Time.deltaTime;

        // Decide between 2D-style (X/Y) and 3D-style (X/Z) panning.
        // If the camera looks mostly along world Z (typical 2D), pan in X/Y.
        // Otherwise pan on the ground plane (X/Z) relative to camera yaw.
        bool panInXY = false;
        if (cachedCamera != null)
        {
            float zAlignment = Mathf.Abs(Vector3.Dot(cachedCamera.transform.forward, Vector3.forward));
            panInXY = zAlignment > 0.9f;
        }
        else
        {
            float zAlignment = Mathf.Abs(Vector3.Dot(cameraTransform.forward, Vector3.forward));
            panInXY = zAlignment > 0.9f;
        }

        Vector3 pos = cameraTransform.position;

        if (panInXY)
        {
            // 2D / side view: screen right = world +X, screen up = world +Y
            pos.x += moveX * step;
            pos.y += moveY * step;
        }
        else
        {
            // 3D / top-down / angled: move on XZ plane, camera-relative.
            Vector3 forward = cameraTransform.forward;
            forward.y = 0f;
            if (forward.sqrMagnitude < 0.0001f)
                forward = Vector3.forward;
            forward.Normalize();

            Vector3 right = cameraTransform.right;
            right.y = 0f;
            if (right.sqrMagnitude < 0.0001f)
                right = Vector3.right;
            right.Normalize();

            pos += right * (moveX * step);
            pos += forward * (moveY * step);
        }

        if (clampCamera)
        {
            pos.x = Mathf.Clamp(pos.x, minBounds.x, maxBounds.x);
            pos.y = Mathf.Clamp(pos.y, minBounds.y, maxBounds.y);
            pos.z = Mathf.Clamp(pos.z, minBounds.z, maxBounds.z);
        }

        cameraTransform.position = pos;
    }

    void OnValidate()
    {
        if (panSpeed < 0f)
            panSpeed = 0f;
        if (borderThickness < 1f)
            borderThickness = 1f;
        if (minBounds.x > maxBounds.x || minBounds.y > maxBounds.y || minBounds.z > maxBounds.z)
        {
            Vector3 newMin = Vector3.Min(minBounds, maxBounds);
            Vector3 newMax = Vector3.Max(minBounds, maxBounds);
            minBounds = newMin;
            maxBounds = newMax;
        }
    }
}

