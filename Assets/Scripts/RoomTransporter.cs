using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.SceneManagement;

public class RoomTransporter : MonoBehaviour
{
    public string ID;
    public string toLoad;
    public string transporterTarget;

    // Set by PersistentSceneBootstrap - scene name that must never be unloaded.
    public static string PersistentSceneName = "";

    public int doorToOpen;

    public Vector2 MoveTarget;
    public GameObject[] doors;

    public bool transportDebug = false;
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

    private System.Collections.IEnumerator TransportRoutine()
    {
        // Capture current scene before moving to DDOL
        // This is the level scene we want to replace. The persistent scene
        // (RoomTransporter.PersistentSceneName) is left loaded.
        Scene currentScene = gameObject.scene;

        // 2. get into the Don't destroy on load scene
        DontDestroyOnLoad(gameObject);

        // 3. unloads the current scene - but NEVER the persistent scene
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

        // 4. teleports itself to move target
        transform.position = new Vector3(MoveTarget.x, MoveTarget.y, transform.position.z);

        // 5. loads the toLoad scene ADDITIVELY so the persistent scene survives.
        // (LoadSceneMode.Single would unload EVERYTHING including the persistent scene.)
        AsyncOperation load = SceneManager.LoadSceneAsync(toLoad, LoadSceneMode.Additive);
        while (load != null && !load.isDone)
        {
            yield return null;
        }
        // Let the new scene finish initialising (Awake/Start of its objects)
        yield return null;

        // Make the newly loaded level the ActiveScene so it behaves like
        // the old Single-load flow (lighting, instantiation target, etc.).
        // The persistent scene stays loaded in the background.
        Scene newScene = SceneManager.GetSceneByName(toLoad);
        if (newScene.IsValid() && newScene.isLoaded)
        {
            SceneManager.SetActiveScene(newScene);
        }

        // 6. Finds the target transporter
        RoomTransporter target = null;
        RoomTransporter[] all = FindObjectsByType<RoomTransporter>(FindObjectsInactive.Exclude);
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
            // Still move into the new level so we are not stranded in DDOL.
            Scene fallback = SceneManager.GetSceneByName(toLoad);
            if (fallback.IsValid() && fallback.isLoaded)
            {
                SceneManager.MoveGameObjectToScene(gameObject, fallback);
            }
            yield break;
        }

        // Move ourselves into the newly loaded scene so we truly replace the target
        SceneManager.MoveGameObjectToScene(gameObject, target.gameObject.scene);

        // 7. teleports itself to it's location, copies it's variables into itself
        transform.position = target.transform.position;
        transform.rotation = target.transform.rotation;

        ID = target.ID;
        toLoad = target.toLoad;
        transporterTarget = target.transporterTarget;
        doorToOpen = target.doorToOpen;
        MoveTarget = target.MoveTarget;
        doors = target.doors;

        // 8. destroys the target transporter (not itself)
        Destroy(target.gameObject);
    }
    // Update is called once per frame
    void Update()
    {
        if(transportDebug)
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
