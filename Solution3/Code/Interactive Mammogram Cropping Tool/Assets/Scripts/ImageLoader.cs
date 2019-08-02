using SFB;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using UnityEngine.UI;
using UnityEngine.EventSystems;

public class ImageLoader : MonoBehaviour
{
    private string OriginalImageFolder, MaskImageFolder;
    private string[] OriginalImages, MaskImages;
    private Dictionary<string, Texture2D> OriginalImageTextures = new Dictionary<string, Texture2D>();
    private Dictionary<string, Texture2D> MaskImageTextures = new Dictionary<string, Texture2D>();
    private Dictionary<string, Texture2D> MaskImageWithAlphaTextures = new Dictionary<string, Texture2D>();
    private List<string> ImageNames = new List<string>();
    public GameObject MaskTexture, MammoTexture, MaskAlphaTexture;
    public GameObject Pen, MirroredPen;
    public GameObject NextImageButton, LastImageButton, SaveImageButton;
    public GameObject Canvas, EventSystem;

    private bool mammoImagesLoaded = false;
    private bool mammoMasksLoaded = false;

    private int currentIndex = -1;

    private string paintingTool = "Rectangle";
    private int width = 20;
    private bool erasing = false;

    private Texture2D PenTexture;

    private Texture2D currentMaskTexture, currentAlphaMaskTexture;

    GraphicRaycaster m_Raycaster;
    PointerEventData m_PointerEventData;
    EventSystem m_EventSystem;

    public void Start()
    {
        m_Raycaster = Canvas.GetComponent<GraphicRaycaster>();
        m_EventSystem = EventSystem.GetComponent<EventSystem>();

        NextImageButton.GetComponent<Button>().interactable = false;
        LastImageButton.GetComponent<Button>().interactable = false;

        UseRectanglePen();
        Pen.SetActive(false);
        MirroredPen.SetActive(false);
    }
    public void Update()
    {
        m_PointerEventData = new PointerEventData(m_EventSystem);
        m_PointerEventData.position = Input.mousePosition;

        List<RaycastResult> results = new List<RaycastResult>();

        m_Raycaster.Raycast(m_PointerEventData, results);

        bool hasHit = false;
        if (mammoImagesLoaded && mammoMasksLoaded)
        {
            foreach (RaycastResult result in results)
            {
                if (result.gameObject.name == MaskTexture.name)
                {
                    Pen.SetActive(true);
                    MirroredPen.SetActive(true);
                    Vector2 relativePoint = ScreenPointToMaskCoordinate(result.screenPosition);
                    Pen.transform.position = result.screenPosition;
                    MirroredPen.transform.position = GetMirroredPointForMammo(relativePoint);
                    if (Input.GetMouseButton(0))
                    {
                        PaintAtCoord(relativePoint);
                    }
                    hasHit = true;
                }
                else if (result.gameObject.name == MammoTexture.name)
                {
                    Pen.SetActive(true);
                    MirroredPen.SetActive(true);
                    Vector2 relativePoint = ScreenPointToMammoCoordinate(result.screenPosition);
                    Pen.transform.position = result.screenPosition;
                    MirroredPen.transform.position = GetMirroredPointForMask(relativePoint);
                    if (Input.GetMouseButton(0))
                    {
                        PaintAtCoord(relativePoint);
                    }
                    hasHit = true;
                }
            }
            if (!hasHit)
            {
                Pen.SetActive(false);
                MirroredPen.SetActive(false);
            }
        }
    }
    private Vector2 ScreenPointToMammoCoordinate(Vector2 screenPoint)
    {
        Vector3[] v = new Vector3[4];
        MammoTexture.GetComponent<RectTransform>().GetWorldCorners(v);
        Vector2 ans = new Vector2(screenPoint.x, screenPoint.y);
        ans = ans - new Vector2(v[0].x, v[0].y);
        return ans;
    }
    private Vector2 ScreenPointToMaskCoordinate(Vector2 screenPoint)
    {
        Vector3[] v = new Vector3[4];
        MaskTexture.GetComponent<RectTransform>().GetWorldCorners(v);
        Vector2 ans = new Vector2(screenPoint.x, screenPoint.y);
        ans = ans - new Vector2(v[0].x, v[0].y);
        return ans;
    }
    private Vector2 GetMirroredPointForMammo(Vector2 relativePoint)
    {
        Vector3[] v = new Vector3[4];
        MammoTexture.GetComponent<RectTransform>().GetWorldCorners(v);
        return relativePoint + new Vector2(v[0].x, v[0].y);
    }
    private Vector2 GetMirroredPointForMask(Vector2 relativePoint)
    {
        Vector3[] v = new Vector3[4];
        MaskTexture.GetComponent<RectTransform>().GetWorldCorners(v);
        return relativePoint + new Vector2(v[0].x, v[0].y);
    }
    private void PaintAtCoord(Vector2 coord)
    {

        Texture2D maskTexture = currentMaskTexture;
        Texture2D maskAlphaTexture = currentAlphaMaskTexture;
        
        Vector2 scaleForCoordinates = new Vector2(maskTexture.width / MaskTexture.GetComponent<RectTransform>().sizeDelta.x,
                maskTexture.height / MaskTexture.GetComponent<RectTransform>().sizeDelta.y);
        coord = new Vector2(coord.x * scaleForCoordinates.x, coord.y * scaleForCoordinates.y);

        float widthX = width * scaleForCoordinates.x;
        float widthY = width * scaleForCoordinates.y;
        
        
        if (paintingTool == "Rectangle")
        {
            int xStart = (int)Mathf.Min(Mathf.Max(0, (coord.x - widthX / 2f)), maskTexture.width);
            int xEnd = (int)Mathf.Min(maskTexture.width, xStart + widthX);
            int yStart = (int)Mathf.Min(Mathf.Max(0, (coord.y - widthY / 2f)), maskTexture.height);
            int yEnd = (int)Mathf.Min(maskTexture.height, yStart + widthY);

            Color[] colors = new Color[(int)((xEnd-xStart) * (yEnd-yStart))];
            Color[] colorsAlpha = new Color[(int)((xEnd - xStart) * (yEnd - yStart))];
            for(int i = 0; i < colors.Length; i++)
            {
                if (erasing)
                {
                    colors[i] = new Color(1, 1, 1);
                    colorsAlpha[i] = new Color(0, 0, 0, 0);
                }
                else
                {
                    colors[i] = new Color(0, 0, 0);
                    colorsAlpha[i] = new Color(0, 0, 0, 1);
                }
            }
            maskTexture.SetPixels(xStart, yStart, (xEnd - xStart), (yEnd - yStart), colors);
            maskAlphaTexture.SetPixels(xStart, yStart, (xEnd - xStart), (yEnd - yStart), colorsAlpha);
        }
        else if(paintingTool == "SoftCircle")
        {         
            for(int x = (int)Mathf.Max(0, coord.x - widthX / 2f); x < (int)Mathf.Min(maskTexture.width, coord.x + widthX / 2f); x++)
            {
                for (int y = (int)Mathf.Max(0, coord.y - widthY / 2f); y < (int)Mathf.Min(maskTexture.height, coord.y + widthY / 2f); y++)
                {
                    if (Vector2.Distance(coord, new Vector2(x, y)) <= Vector2.Distance(coord, new Vector2(coord.x - widthX / 2f, coord.y)))
                    {
                        Color c = Color.white;
                        Color c_alpha = new Color(0, 0, 0, 1);
                        if (erasing)
                        {
                            c = c * Vector2.Distance(coord, new Vector2(x, y)) / Vector2.Distance(coord, new Vector2(coord.x - widthX / 2f, coord.y - widthY / 2f));
                            c_alpha = c_alpha * (1 - Vector2.Distance(coord, new Vector2(x, y)) / Vector2.Distance(coord, new Vector2(coord.x - widthX / 2f, coord.y - widthY / 2f)));
                        }
                        else
                        {
                            c = c * (1 - Vector2.Distance(coord, new Vector2(x, y)) / Vector2.Distance(coord, new Vector2(coord.x - widthX / 2f, coord.y - widthY / 2f)));
                            c_alpha = c_alpha * Vector2.Distance(coord, new Vector2(x, y)) / Vector2.Distance(coord, new Vector2(coord.x - widthX / 2f, coord.y - widthY / 2f));

                        }
                        maskTexture.SetPixel(x, y, c);
                        maskAlphaTexture.SetPixel(x, y, c_alpha);
                    }
                }
            }            
        }
        else if(paintingTool == "Circle")
        {
            for (int x = (int)Mathf.Max(0, coord.x - widthX / 2f); x < (int)Mathf.Min(maskTexture.width, coord.x + widthX / 2f); x++)
            {
                for (int y = (int)Mathf.Max(0, coord.y - widthY / 2f); y < (int)Mathf.Min(maskTexture.height, coord.y + widthY / 2f); y++)
                {
                    if (Vector2.Distance(coord, new Vector2(x, y)) <= Vector2.Distance(coord, new Vector2(coord.x - widthX / 2f, coord.y))) { 
                        Color c = Color.white;
                        Color c_alpha = new Color(0, 0, 0, 1);
                        if (!erasing)
                        {
                            c = Color.black;
                        }
                        if (erasing)
                        {
                            c_alpha = Color.clear;
                        }
                        maskTexture.SetPixel(x, y, c);
                        maskAlphaTexture.SetPixel(x, y, c_alpha);
                    }
                }
            }
        }
        
        maskTexture.Apply();
        maskAlphaTexture.Apply();

        currentMaskTexture = maskTexture;
        currentAlphaMaskTexture = maskAlphaTexture;

        MaskTexture.GetComponent<RawImage>().texture = maskTexture;
        MaskAlphaTexture.GetComponent<RawImage>().texture = maskAlphaTexture;
        
    }
    public void ChooseOriginalImageFolder()
    {
        OriginalImageTextures = new Dictionary<string, Texture2D>();
        string[] path = StandaloneFileBrowser.OpenFolderPanel("Open Folder", "", false);
        if(path.Length > 0)
        {
            OriginalImageFolder = path[0];
            print(path[0] + " selected as original image folder.");
            OriginalImages = Directory.GetFiles(OriginalImageFolder);
            print(OriginalImages.Length + " images found.");
            if(OriginalImages.Length > 0)
            {
                foreach(string p in OriginalImages)
                {
                    string[] im_path_split = p.Split(Path.DirectorySeparatorChar);
                    string im_name = im_path_split[im_path_split.Length - 1];
                    Texture2D tex = null;
                    byte[] fileData;
                    if (File.Exists(p))
                    {
                        fileData = File.ReadAllBytes(p);
                        tex = new Texture2D(2, 2);
                        tex.LoadImage(fileData);
                        if (OriginalImageTextures.ContainsKey(im_name)){
                            print("Duplicate image detected. OriginalImageTextures already contains " + im_name);
                        }
                        else {
                            OriginalImageTextures.Add(im_name, tex);
                            if (!ImageNames.Contains(im_name))
                            {
                                ImageNames.Add(im_name);
                            }
                        }
                    }
                }
                mammoImagesLoaded = true;
                if (mammoImagesLoaded && mammoMasksLoaded)
                {
                    NextImageButton.GetComponent<Button>().interactable = true;
                    currentIndex = 0;
                    LoadAtIndex(currentIndex);
                }
            }
            else
            {
                print("No images found");
            }
        }
        else{
            print("No path chosen");
        }
    }
    public void ChooseMaskImageFolder()
    {
        MaskImageTextures = new Dictionary<string, Texture2D>();
        MaskImageWithAlphaTextures = new Dictionary<string, Texture2D>();
        string[] path = StandaloneFileBrowser.OpenFolderPanel("Open Folder", "", false);
        if (path.Length > 0)
        {
            OriginalImageFolder = path[0];
            print(path[0] + " selected as original image folder.");
            OriginalImages = Directory.GetFiles(OriginalImageFolder);
            print(OriginalImages.Length + " images found.");
            if (OriginalImages.Length > 0)
            {
                foreach (string p in OriginalImages)
                {
                    string[] im_path_split = p.Split(Path.DirectorySeparatorChar);
                    string im_name = im_path_split[im_path_split.Length - 1];
                    Texture2D tex = null, texWithAlpha = null;
                    byte[] fileData;
                    if (File.Exists(p))
                    {
                        fileData = File.ReadAllBytes(p);
                        tex = new Texture2D(2, 2);
                        texWithAlpha = new Texture2D(2, 2);
                        tex.LoadImage(fileData);
                        texWithAlpha.LoadImage(fileData);
                        if (MaskImageTextures.ContainsKey(im_name))
                        {
                            print("Duplicate image detected. OriginalImageTextures already contains " + im_name);
                        }
                        else
                        {
                            MaskImageTextures.Add(im_name, tex);
                            for(int x = 0; x < texWithAlpha.width; x++)
                            {
                                for(int y = 0; y < texWithAlpha.height; y++)
                                {
                                    if(texWithAlpha.GetPixel(x, y).r > 0.5f)
                                    {
                                        texWithAlpha.SetPixel(x, y, Color.clear);
                                    }
                                }
                            }
                            texWithAlpha.Apply();
                            MaskImageWithAlphaTextures.Add(im_name, texWithAlpha);
                            if (!ImageNames.Contains(im_name))
                            {
                                ImageNames.Add(im_name);
                            }
                        }
                    }
                }
                mammoMasksLoaded = true;
                if(mammoImagesLoaded && mammoMasksLoaded)
                {
                    NextImageButton.GetComponent<Button>().interactable = true;
                    currentIndex = 0;
                    LoadAtIndex(currentIndex);
                }
            }
            else
            {
                print("No images found");
            }
        }
        else
        {
            print("No path chosen");
        }
    }
    public void LastImage()
    {
        if(mammoMasksLoaded && mammoImagesLoaded && currentIndex > 0)
        {
            currentIndex--;
            LoadAtIndex(currentIndex);
            if (currentIndex == 0)
            {
                LastImageButton.GetComponent<Button>().interactable = false;
            }
            NextImageButton.GetComponent<Button>().interactable = true;
            SaveImageButton.GetComponent<Button>().interactable = true;
        }
    }
    public void NextImage()
    {
        if(mammoMasksLoaded && mammoImagesLoaded && currentIndex < OriginalImageTextures.Count - 1)
        {
            currentIndex++;
            LoadAtIndex(currentIndex);
            if(currentIndex == OriginalImageTextures.Count - 1)
            {
                NextImageButton.GetComponent<Button>().interactable = false;
            }
            LastImageButton.GetComponent<Button>().interactable = true;
            SaveImageButton.GetComponent<Button>().interactable = true;
        }
    }
    public void SaveImage()
    {
        Texture2D mammoTexture = (Texture2D)MammoTexture.GetComponent<RawImage>().texture;
        Texture2D maskTexture = (Texture2D)MaskTexture.GetComponent<RawImage>().texture;
        Texture2D finalTexture = new Texture2D(mammoTexture.width, mammoTexture.height);

        for (int x = 0; x < finalTexture.width; x++)
        {
            for (int y = 0; y < finalTexture.height; y++)
            {
                float c = mammoTexture.GetPixel(x, y).r;
                float a = maskTexture.GetPixel(x, y).r;
                finalTexture.SetPixel(x, y, new Color(c * a, c * a, c * a));
            }
        }
        byte[] finalBytes = finalTexture.EncodeToPNG();
        // Save file with filter
        var extensionList = new[] {
            new ExtensionFilter("Image", "png"),
        };
        string path = StandaloneFileBrowser.SaveFilePanel("Save File", "", ImageNames[currentIndex], extensionList);
        if(path != null && path != "")
        {
            File.WriteAllBytes(path, finalBytes);
        }
    }
    public void UseCirclePen()
    {
        paintingTool = "Circle";
        PenTexture = new Texture2D(width, width);
        Color[] c = new Color[(int)(width * width)];
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < width; y++)
            {
                if (erasing)
                {
                    if (Vector2.Distance(new Vector2(x, y), new Vector2(width / 2, width / 2)) < Vector2.Distance(new Vector2(0, width / 2f), new Vector2(width / 2f, width / 2f))) {
                        PenTexture.SetPixel(x, y, Color.white);
                    }
                    else
                    {
                        PenTexture.SetPixel(x, y, Color.clear);
                    }
                }
                else
                {
                    if (Vector2.Distance(new Vector2(x, y), new Vector2(width / 2, width / 2)) < Vector2.Distance(new Vector2(0, width / 2f), new Vector2(width / 2f, width / 2f)))
                    {
                        PenTexture.SetPixel(x, y, Color.black);
                    }
                    else
                    {
                        PenTexture.SetPixel(x, y, Color.clear);
                    }
                }
            }
        }
        PenTexture.Apply();
        Pen.GetComponent<RawImage>().texture = PenTexture;
        MirroredPen.GetComponent<RawImage>().texture = PenTexture;
    }
    public void UseSoftCirclePen()
    {
        paintingTool = "SoftCircle";
        PenTexture = new Texture2D(width, width);
        Color[] c = new Color[(int)(width * width)];
        for (int x = 0; x < width; x++)
        {
            for(int y = 0; y < width; y++)
            {
                if (erasing)
                {
                    PenTexture.SetPixel(x, y, Color.white * (1 - (Vector2.Distance(new Vector2(x, y), new Vector2(width / 2, width / 2)) / Vector2.Distance(Vector2.zero, new Vector2(width / 2f, width / 2f)))));
                }
                else
                {
                    PenTexture.SetPixel(x, y, Color.black * (1- (Vector2.Distance(new Vector2(x, y), new Vector2(width / 2, width / 2)) / Vector2.Distance(Vector2.zero, new Vector2(width / 2f, width / 2f)))));
                }
            }
        }
        PenTexture.Apply();
        Pen.GetComponent<RawImage>().texture = PenTexture;
        MirroredPen.GetComponent<RawImage>().texture = PenTexture;
    }
    public void UseRectanglePen()
    {
        paintingTool = "Rectangle";

        PenTexture = new Texture2D(width, width);
        Color[] c = new Color[(int)(width * width)];
        for (int i = 0; i < c.Length; i++)
        {
            if (erasing)
            {
                c[i] = Color.white;
            }
            else
            {
                c[i] = Color.black;
            }
        }
        PenTexture.SetPixels(c);
        PenTexture.Apply();
        Pen.GetComponent<RawImage>().texture = PenTexture;
        MirroredPen.GetComponent<RawImage>().texture = PenTexture;
    }
    public void SetDrawing()
    {
        erasing = false;
        if (paintingTool == "Rectangle")
        {
            UseRectanglePen();
        }
        else if (paintingTool == "SoftCircle")
        {
            UseSoftCirclePen();
        }
        else if (paintingTool == "Circle")
        {
            UseCirclePen();
        }
    }
    public void SetErasing()
    {
        erasing = true;
        if (paintingTool == "Rectangle")
        {
            UseRectanglePen();
        }
        else if (paintingTool == "SoftCircle")
        {
            UseSoftCirclePen();
        }
        else if (paintingTool == "Circle")
        {
            UseCirclePen();
        }
    }
    public void ChangeOpacityOfMask(float value)
    {
        MaskAlphaTexture.GetComponent<RawImage>().color = new Color(1, 1, 1, value);
    }
    public void ChangeSizeOfBrush(float value)
    {
        width = (int)value;
        Pen.GetComponent<RectTransform>().sizeDelta = new Vector2(value, value);
        MirroredPen.GetComponent<RectTransform>().sizeDelta = new Vector2(value, value);
        if(paintingTool == "Rectangle")
        {
            UseRectanglePen();
        }
        else if(paintingTool == "SoftCircle")
        {
            UseSoftCirclePen();
        }
        else if(paintingTool == "Circle")
        {
            UseCirclePen();
        }
    }
    private void LoadAtIndex(int index)
    {
        MammoTexture.GetComponent<RawImage>().texture = OriginalImageTextures[ImageNames[index]];

        MaskTexture.GetComponent<RawImage>().texture = MaskImageTextures[ImageNames[index]];
        currentMaskTexture = MaskImageTextures[ImageNames[index]];
        MaskAlphaTexture.GetComponent<RawImage>().texture = MaskImageWithAlphaTextures[ImageNames[index]];
        currentAlphaMaskTexture = MaskImageWithAlphaTextures[ImageNames[index]];
    }

    public static Texture2D ToTexture2D(Texture texture)
    {
        return Texture2D.CreateExternalTexture(
            texture.width,
            texture.height,
            TextureFormat.RGB24,
            false, false,
            texture.GetNativeTexturePtr());
    }
}
