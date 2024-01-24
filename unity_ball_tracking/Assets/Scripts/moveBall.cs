using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System;

[System.Serializable] // This makes the class visible in the inspector
public class Ball_Detection
{
    public bool is_detected;
    public float x;
    public float y;
    public float z;
    public float radius;
    public float speed;

    public static Ball_Detection CreateFromJSON(string jsonString)
    {
        return JsonUtility.FromJson<Ball_Detection>(jsonString);
    }
}


public class moveBall : MonoBehaviour
{
    int port;
    Thread receiveThread;
    UdpClient client;
    Ball_Detection ball_detection;
    
    private int count;
    int highScore = 0;
    private int lives;
    private float scrollSpeed = 0.5f;
    private ShowAttributes showAttributes;
    private float actualSpeed;

    public GameOverScreen GameOverScreen;


    [Header("Movement Controls")]
    [SerializeField] 
    [Range(0.001f, 0.01f)]
    private float mvmtFactor = 0.005f;

    // Start is called before the first frame update
    private void Start()
    {
        port = 5065;
        InitUDP();
        showAttributes = FindObjectOfType<ShowAttributes>();
        count = 0;
        lives = 3;

        highScore = PlayerPrefs.GetInt("HighScore", 0);
    }

    // Update is called once per frame
    private void Update()
    {
        if (ball_detection.is_detected)
        {
            float factor = - mvmtFactor; 
            transform.position = new Vector3(ball_detection.x*factor, 
                                                ball_detection.y*factor, 
                                                ball_detection.z*factor);
            actualSpeed = ball_detection.speed;
        }
        // Print the different attributes in the game
        showAttributes.Update(actualSpeed, count, lives);
    }

    public void GameOver()
    {
        GameOverScreen.Setup(count);

    }

    public void Reset()
    {
        count = 0;
        lives = 3;
        scrollSpeed = 0.5f;
        actualSpeed = 0.0f;

        FindObjectOfType<ManagePickUp>().Reset();
    }

    public float getScrollSpeed()
    {
        return scrollSpeed;
    }

    public int getLives()
    {
        return lives;
    }

    public void NotCollected()
    {
        lives -= 1;
        if (lives == 0)
        {
            GameOver();
        }

    }
    private void OnTriggerEnter(Collider other)
    // When the player touches a pick up
    {
        if (other.gameObject.CompareTag("PickUp"))
        {
            count += 1;
            scrollSpeed += 0.1f;
            // Debug.Log("Scroll speed increased to " + scrollSpeed);
            FindObjectOfType<ManagePickUp>().SpawnNextPickUp();
            other.gameObject.SetActive(false);

            if (highScore < count)
            {
                PlayerPrefs.SetInt("HighScore", count);
            }
        }
    }


    private void InitUDP()
    {
        print("UDP Initialized");

        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();
    }

    private void ReceiveData()
    {
        client = new UdpClient(port);
        while (true)
        {
            try
            {
                IPEndPoint anyIP = new IPEndPoint(IPAddress.Parse("0.0.0.0"), port);
                byte[] data = client.Receive(ref anyIP);

                string text = Encoding.UTF8.GetString(data);
                // print (">> " + text);
                // Debug.Log(text);
                ball_detection = Ball_Detection.CreateFromJSON(text);

                // Debug.Log(ball_detection.speed);

            }

            catch(Exception e)
            {
                print (e.ToString()); //7
            }
        }
    }
                
}
