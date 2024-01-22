using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;

public class GameOverScreen : MonoBehaviour
{
    public Text pointsText;
    public Text HighScoreText; 
    
    public void Setup(int score)
    {
        gameObject.SetActive(true);
        pointsText.text = score.ToString() + " POINTS";
        int highScore = PlayerPrefs.GetInt("HighScore", 0);
        HighScoreText.text = "High Score: " + highScore.ToString();
    }

    public void RestartButton()
    {
        gameObject.SetActive(false);
        // reset score and lives
        FindObjectOfType<moveBall>().Reset();
    }

    public void ExitButton()
    {
        SceneManager.LoadScene("MainMenu");
    }

}
