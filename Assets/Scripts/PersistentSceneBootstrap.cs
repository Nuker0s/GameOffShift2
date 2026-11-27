using UnityEngine;
using UnityEngine.SceneManagement;

/// <summary>
/// Place this component in every level scene that needs the persistent scene.
/// On Awake it ensures the persistent scene is loaded additively (only once),
/// and registers its name in RoomTransporter.PersistentSceneName so the
/// transporter never unloads it.
/// </summary>
public class PersistentSceneBootstrap : MonoBehaviour
{
    [Tooltip("Scene name (as in Build Settings) of the persistent scene to keep loaded.")]
    public string persistentSceneName = "Persistent";

    // Static guard so two bootstraps awakening in the same frame don't double-load.
    private static bool s_IsLoadingPersistent = false;

    private void Awake()
    {
        if (string.IsNullOrEmpty(persistentSceneName))
        {
            return;
        }

        // Tell the transporter which scene to preserve.
        RoomTransporter.PersistentSceneName = persistentSceneName;

        // If this bootstrap itself lives inside the persistent scene, don't try to load it again.
        if (gameObject.scene.IsValid() && gameObject.scene.name == persistentSceneName)
        {
            return;
        }

        Scene existing = SceneManager.GetSceneByName(persistentSceneName);
        if (existing.IsValid() && existing.isLoaded)
        {
            return;
        }

        if (s_IsLoadingPersistent)
        {
            return;
        }

        s_IsLoadingPersistent = true;
        StartCoroutine(LoadPersistentRoutine());
    }

    private System.Collections.IEnumerator LoadPersistentRoutine()
    {
        AsyncOperation load = SceneManager.LoadSceneAsync(persistentSceneName, LoadSceneMode.Additive);
        if (load == null)
        {
            Debug.LogError("[PersistentSceneBootstrap] Could not start loading persistent scene '" + persistentSceneName + "'. Is it added to Build Settings?");
            s_IsLoadingPersistent = false;
            yield break;
        }

        while (!load.isDone)
        {
            yield return null;
        }

        s_IsLoadingPersistent = false;

        Scene persistent = SceneManager.GetSceneByName(persistentSceneName);
        if (!persistent.IsValid() || !persistent.isLoaded)
        {
            Debug.LogError("[PersistentSceneBootstrap] Failed to load persistent scene '" + persistentSceneName + "'.");
        }
        // NOTE: we intentionally leave the current level as ActiveScene.
        // The persistent scene stays loaded in the background and RoomTransporter
        // preserves it by only unloading gameObject.scene and only loading Additive.
    }
}
