using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ManagePickUp : MonoBehaviour
{
    public GameObject pickUp;


    [Header("Spawn Area")]
    [SerializeField]
    public Vector3 center;
    [SerializeField]
    public Vector3 size;

    // Update is called once per frame
    void Update()
    {
        transform.Rotate(new Vector3(15, 30, 45) * Time.deltaTime);
        float scrollSpeed = FindObjectOfType<moveBall>().getScrollSpeed();
        // Translate it in the x direction relative to the world coordinate system by taking into account the above rotation 
        transform.Translate(Vector3.right * Time.deltaTime * scrollSpeed, 
                            Space.World);
    
        if (transform.position.x > 2)
        {
            // Debug.Log("Pick up not collected");
            FindObjectOfType<moveBall>().NotCollected();
            SpawnNextPickUp();
            pickUp.SetActive(false);

            
        }
    }

    public void Reset()
    {
        transform.position = center;
        SpawnNextPickUp();
        pickUp.SetActive(false);
    }

    public void SpawnNextPickUp()
    {
        Vector3 pos = center + new Vector3(Random.Range(-size.x / 2, size.x / 2), 
                                            Random.Range(-size.y / 2, size.y / 2),
                                            Random.Range(-size.z / 2, size.z / 2));

        Instantiate(pickUp, pos, Quaternion.identity);

    }


}
