using UnityEngine;

/// <summary>
/// Scene-level settings for rendering ParticleSolidifier output.
/// Place one instance in every scene that can solidify particles.
/// </summary>
[DefaultExecutionOrder(-1000)]
public sealed class SolidifiedParticleRenderConfig : MonoBehaviour
{
    public static SolidifiedParticleRenderConfig Instance { get; private set; }

    [Header("Rendering")]
    [SerializeField] private bool renderEnabled = true;
    [SerializeField, Tooltip("Must read vertex colors. For URP 2D lighting, use a Sprite Lit Shader Graph with Sort 3D As 2D Compatible enabled and multiply a Vertex Color node into Base Color.")]
    private Material renderMaterial;
    [SerializeField] private string sortingLayerName = "Default";
    [SerializeField] private int sortingOrder;
    [SerializeField, Range(0, 31)] private int gameObjectLayer;
    [SerializeField] private float renderZOffset;
    [SerializeField, Tooltip("Optional prefab containing ShadowCaster2D and SolidifiedBodyShadowData. Set its Casting Source to Solidified Particle Outline.")]
    private GameObject shadowCasterPrefab;

    [Header("Particle Colors")]
    [SerializeField] private Color baseColor = new Color(0.46f, 0.74f, 0.65f, 1f);
    [SerializeField] private bool randomizeParticleColors = true;
    [SerializeField, Range(0f, 0.5f), Tooltip("Hue variation is normalized: 0.05 is 5% of the hue wheel, or 18 degrees.")]
    private float hueVariation = 0.015f;
    [SerializeField, Range(0f, 1f)] private float saturationVariation = 0.08f;
    [SerializeField, Range(0f, 1f)] private float brightnessVariation = 0.10f;

    public bool RenderSolidifiedBodies => renderEnabled;
    public Material RenderMaterial => renderMaterial;
    public string SortingLayerName => sortingLayerName;
    public int SortingOrder => sortingOrder;
    public int GameObjectLayer => gameObjectLayer;
    public float RenderZOffset => renderZOffset;
    public GameObject ShadowCasterPrefab => shadowCasterPrefab;

    private void OnEnable()
    {
        if (Instance != null && Instance != this)
        {
            Debug.LogError("[SolidifiedParticleRenderConfig] More than one config exists in the scene. The duplicate was disabled.", this);
            enabled = false;
            return;
        }

        Instance = this;
    }

    private void OnDisable()
    {
        if (Instance == this)
            Instance = null;
    }

    /// <summary>
    /// Generates one color for one original particle. Call once per snapshot,
    /// not once per polygon fragment, so all fragments keep the same color.
    /// </summary>
    public Color GenerateParticleColor()
    {
        if (!randomizeParticleColors)
            return baseColor;

        Color.RGBToHSV(baseColor, out float baseHue, out float baseSaturation, out float baseValue);

        float hue = Mathf.Repeat(
            baseHue + Random.Range(-hueVariation, hueVariation),
            1f);

        float saturation = Mathf.Clamp01(
            baseSaturation + Random.Range(-saturationVariation, saturationVariation));

        float value = Mathf.Clamp01(
            baseValue + Random.Range(-brightnessVariation, brightnessVariation));

        Color generatedColor = Color.HSVToRGB(hue, saturation, value);
        generatedColor.a = baseColor.a;
        return generatedColor;
    }
}
