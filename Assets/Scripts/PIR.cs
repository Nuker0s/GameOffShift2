using Unity.Scripting.LifecycleManagement;
using UnityEngine;
using UnityEngine.InputSystem;

public class PIR : MonoBehaviour
{
    public PlayerInput playerInput;
    public static PIR instance;
    public InputAction move;
    public InputAction stateChange;

    void Awake()
    {
        playerInput = GetComponent<PlayerInput>();
        instance = this;
    }
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        move = playerInput.actions["Move"];
        stateChange = playerInput.actions["Interact"];
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
