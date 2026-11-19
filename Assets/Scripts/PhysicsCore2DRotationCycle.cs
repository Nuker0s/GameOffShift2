
using System.Collections.Generic;
using UnityEngine;
using Unity.U2D.Physics;

[DisallowMultipleComponent]
public class PhysicsCore2DRotationCycle : MonoBehaviour
{
    public enum RotationMode
    {
        KinematicVelocity,
        TorquePID
    }

    [Header("Rotation")]
    public RotationMode mode = RotationMode.KinematicVelocity;

    [Tooltip("X = degrees, Y = seconds. Negative Y means hold.")]
    public List<Vector2> cycle = new List<Vector2>();

    public bool positiveXIsClockwise = true;
    public bool paused;
    public bool automaticFixedUpdate = true;

    [SerializeField] private PhysicsPose physicsPose;

    [Header("Kinematic")]
    [Tooltip("Safety limit for commanded angular velocity, in degrees/s.")]
    public float maxKinematicAngularVelocity = 720f;

    [Header("Torque PID")]
    [Tooltip("Position error gain.")]
    public float positionGain = 4f;

    [Tooltip("Velocity error gain. Also provides rotational damping.")]
    public float velocityGain = 5f;

    public float integralGain = 0.05f;
    public float integralLimit = 500f;

    [Tooltip("Maximum torque in N*m.")]
    public float maxTorque = 50f;

    [Tooltip("Maximum reference angular velocity in degrees/s.")]
    public float maxReferenceAngularVelocity = 720f;

    [Header("Torque Completion")]
    [Tooltip("Maximum angular error allowed before exact correction.")]
    public float torqueSnapAngleTolerance = 0.1f;

    [Tooltip("Maximum angular velocity allowed before exact correction.")]
    public float torqueSnapVelocityTolerance = 0.5f;

    private PhysicsBody body;
    private bool hasBody;

    // Unwrapped rotation tracking.
    private float unwrappedAngle;
    private float lastWrappedAngle;
    private bool hasAngleSample;

    // Cycle state.
    private int stepIndex;
    private float cycleTargetAngle;

    // Current instruction.
    private float stepStartAngle;
    private float stepEndAngle;
    private float stepDelta;
    private float stepDuration;
    private float stepElapsed;

    // PID state.
    private float integralError;

    // Pause state.
    private bool wasPaused;
    private float pausedTargetAngle;

    private bool modeErrorLogged;

    private const float AngleEpsilon = 0.00001f;

    private void Awake()
    {
        if (physicsPose == null)
            physicsPose = GetComponent<PhysicsPose>();
    }

    private void Start()
    {
        if (!hasBody && physicsPose != null)
            SetBody(physicsPose.body);
    }

    public void SetBody(PhysicsBody physicsBody)
    {
        if (!physicsBody.isValid)
        {
            Debug.LogError(
                "PhysicsCore2DRotationCycle: Invalid PhysicsBody.",
                this
            );

            return;
        }

        body = physicsBody;
        hasBody = true;

        hasAngleSample = false;

        SampleCurrentAngle();
        RestartCycle();
    }

    public void RestartCycle()
    {
        if (!hasBody)
            return;

        SampleCurrentAngle();

        body.angularVelocity = 0f;

        stepIndex = 0;
        cycleTargetAngle = unwrappedAngle;

        integralError = 0f;
        wasPaused = false;

        if (cycle == null || cycle.Count == 0)
            return;

        BeginStep();
    }

    private void FixedUpdate()
    {
        if (automaticFixedUpdate)
            Tick(Time.fixedDeltaTime);
    }

    public void Tick(float deltaTime)
    {
        if (!hasBody || !body.isValid)
            return;

        if (cycle == null || cycle.Count == 0)
            return;

        if (deltaTime <= 0f)
            return;

        if (!ValidateMode())
            return;

        SampleCurrentAngle();

        // PAUSE
        if (paused)
        {
            if (!wasPaused)
            {
                pausedTargetAngle = unwrappedAngle;
                integralError = 0f;
                wasPaused = true;
            }

            if (mode == RotationMode.KinematicVelocity)
            {
                SetExactRotation(pausedTargetAngle);
            }
            else
            {
                // Actively brake and hold the current position.
                ApplyTorquePID(
                    pausedTargetAngle,
                    0f,
                    deltaTime
                );
            }

            // Do not advance instruction time.
            return;
        }

        if (wasPaused)
        {
            wasPaused = false;
            integralError = 0f;
        }

        if (mode == RotationMode.KinematicVelocity)
        {
            TickKinematic(deltaTime);
        }
        else
        {
            TickTorque(deltaTime);
        }
    }

    // --------------------------------------------------
    // STEP MANAGEMENT
    // --------------------------------------------------

    private void BeginStep()
    {
        Vector2 instruction = cycle[stepIndex];

        stepStartAngle = cycleTargetAngle;

        // Negative time means hold.
        if (instruction.y < 0f)
        {
            stepDelta = 0f;
        }
        else
        {
            float direction = positiveXIsClockwise ? -1f : 1f;
            stepDelta = instruction.x * direction;
        }

        stepDuration = Mathf.Abs(instruction.y);
        stepElapsed = 0f;

        stepEndAngle = stepStartAngle + stepDelta;

        integralError = 0f;
    }

    private void CompleteCurrentStep()
    {
        // Exact correction at the destination.
        SetExactRotation(stepEndAngle);

        cycleTargetAngle = stepEndAngle;

        stepIndex++;

        if (stepIndex >= cycle.Count)
            stepIndex = 0;

        BeginStep();
    }

    // --------------------------------------------------
    // KINEMATIC MODE
    // --------------------------------------------------

    private void TickKinematic(float deltaTime)
    {
        // Skip zero-duration instructions.
        int instantStepsProcessed = 0;

        while (stepDuration <= AngleEpsilon &&
               instantStepsProcessed < cycle.Count)
        {
            CompleteCurrentStep();
            instantStepsProcessed++;
        }

        if (stepDuration <= AngleEpsilon)
            return;

        // Directly command the required constant speed.
        // No position-error / deltaTime feedback.
        float commandedVelocity = stepDelta / stepDuration;

        commandedVelocity = Mathf.Clamp(
            commandedVelocity,
            -maxKinematicAngularVelocity,
            maxKinematicAngularVelocity
        );

        body.angularVelocity = commandedVelocity;

        // Advance the instruction timer.
        stepElapsed += deltaTime;

        if (stepElapsed >= stepDuration - AngleEpsilon)
        {
            CompleteCurrentStep();
        }
    }

    // --------------------------------------------------
    // TORQUE MODE
    // --------------------------------------------------

    private void TickTorque(float deltaTime)
    {
        // Advance time only until the suggested duration.
        if (stepDuration > AngleEpsilon)
        {
            stepElapsed = Mathf.Min(
                stepElapsed + deltaTime,
                stepDuration
            );
        }

        float progress = stepDuration <= AngleEpsilon
            ? 1f
            : Mathf.Clamp01(stepElapsed / stepDuration);

        // Moving reference position.
        // Stops at the destination when the suggested time expires.
        float targetAngle = Mathf.LerpUnclamped(
            stepStartAngle,
            stepEndAngle,
            progress
        );

        // Reference speed.
        float nominalAngularVelocity = 0f;

        if (stepDuration > AngleEpsilon &&
            stepElapsed < stepDuration - AngleEpsilon)
        {
            nominalAngularVelocity = stepDelta / stepDuration;
        }

        ApplyTorquePID(
            targetAngle,
            nominalAngularVelocity,
            deltaTime
        );

        bool timeFinished =
            stepDuration <= AngleEpsilon ||
            stepElapsed >= stepDuration - AngleEpsilon;

        // Never complete just because time expired.
        // Wait until the actual body has reached and settled at the target.
        if (timeFinished && IsCloseEnoughToComplete())
        {
            CompleteCurrentStep();
        }
    }

    // --------------------------------------------------
    // TORQUE PID
    // --------------------------------------------------

    private void ApplyTorquePID(
        float targetAngle,
        float nominalAngularVelocity,
        float deltaTime)
    {
        float angleError = targetAngle - unwrappedAngle;

        // The desired velocity follows the suggested trajectory.
        // Position error produces additional acceleration through the
        // proportional controller rather than an unbounded velocity command.
        float desiredAngularVelocity = Mathf.Clamp(
            nominalAngularVelocity,
            -maxReferenceAngularVelocity,
            maxReferenceAngularVelocity
        );

        float velocityError =
            desiredAngularVelocity - body.angularVelocity;

        // Integral.
        float proposedIntegral =
            integralError + angleError * deltaTime;

        proposedIntegral = Mathf.Clamp(
            proposedIntegral,
            -integralLimit,
            integralLimit
        );

        // Desired angular acceleration in degrees/s².
        // Position correction + velocity correction + integral correction.
        float angularAcceleration =
            positionGain * angleError +
            velocityGain * velocityError +
            integralGain * proposedIntegral;

        // Convert angular acceleration into physical torque.
        // PhysicsBody.angularVelocity is degrees/s.
        // Physical torque requires angular acceleration in radians/s².
        float inertia = body.rotationalInertia;

        if (inertia <= AngleEpsilon)
            return;

        float rawTorque =
            inertia *
            angularAcceleration *
            Mathf.Deg2Rad;

        float appliedTorque = Mathf.Clamp(
            rawTorque,
            -maxTorque,
            maxTorque
        );

        // Anti-windup.
        bool saturationWorsensError =
            (rawTorque > maxTorque && angleError > 0f) ||
            (rawTorque < -maxTorque && angleError < 0f);

        if (!saturationWorsensError)
        {
            integralError = proposedIntegral;
        }

        body.ApplyTorque(appliedTorque, true);
    }

    private bool IsCloseEnoughToComplete()
    {
        float angleError = Mathf.Abs(
            stepEndAngle - unwrappedAngle
        );

        float angularVelocity = Mathf.Abs(
            body.angularVelocity
        );

        return angleError <= torqueSnapAngleTolerance &&
               angularVelocity <= torqueSnapVelocityTolerance;
    }

    // --------------------------------------------------
    // ROTATION TRACKING
    // --------------------------------------------------

    private void SampleCurrentAngle()
    {
        float wrappedAngle = body.rotation.degrees;

        if (!hasAngleSample)
        {
            unwrappedAngle = wrappedAngle;
            lastWrappedAngle = wrappedAngle;
            hasAngleSample = true;
            return;
        }

        float difference = Mathf.DeltaAngle(
            lastWrappedAngle,
            wrappedAngle
        );

        unwrappedAngle += difference;
        lastWrappedAngle = wrappedAngle;
    }

    private void SetExactRotation(float targetAngle)
    {
        body.rotation = PhysicsRotate.FromDegrees(targetAngle);
        body.angularVelocity = 0f;

        unwrappedAngle = targetAngle;
        lastWrappedAngle = body.rotation.degrees;
        hasAngleSample = true;
    }

    // --------------------------------------------------
    // VALIDATION
    // --------------------------------------------------

    private bool ValidateMode()
    {
        bool valid = false;

        if (mode == RotationMode.KinematicVelocity)
        {
            valid = body.type == PhysicsBody.BodyType.Kinematic;
        }
        else if (mode == RotationMode.TorquePID)
        {
            valid = body.type == PhysicsBody.BodyType.Dynamic;
        }

        if (!valid && !modeErrorLogged)
        {
            Debug.LogError(
                "PhysicsCore2DRotationCycle: Incorrect PhysicsBody type " +
                "for the selected rotation mode.",
                this
            );

            modeErrorLogged = true;
        }

        if (valid)
            modeErrorLogged = false;

        return valid;
    }
}
