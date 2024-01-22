using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;

public class MainMenuScreen : MonoBehaviour
{
    public Text HighScoreText;

    private void Start()
    {
        gameObject.SetActive(true);
        int highScore = PlayerPrefs.GetInt("HighScore", 0);
        HighScoreText.text = "High Score: " + highScore.ToString();
    }

    public void PlayButton()
    {
        // Load the game scene
        Debug.Log("Play button pressed");
        SceneManager.LoadScene("Game");
    }

    public void ExitButton()
    {
        Application.Quit();
        Debug.Log("Exit button pressed");
    }
}
