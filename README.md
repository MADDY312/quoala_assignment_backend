## Note:
**google_cloud.py** file is the python file that is deployed

## TechStack Used:
### Backend: 
Python(Flask Api), Google Cloud Vision Api(For OCR)
### Frontend:
javascript,React js
### Database:
MongoDb

## Deployment Services:
Backend:**Pythonanywhere**
,Frontend: **Firebase**
,Database:**MongoDB Atlas**

# Journey:
**Intial Plan:** -use Tesseract OCR with python (pytesseract)<br>
             -Use Springboot restcontroller to access the mongodb<br> 
             -use React frontend

**What Changed?**
            -Not used Tesseract OCR:  After a lot of experimenting with the preprocessing and postprocessing of images and trying it a big bunch images which you can see in the souce code and in "**idcardocr.py**" file, the results were still below satisfactory,so I had to switch to Google cloud vision api 
            for the ocr, and the results improved drastically                           
            <br>-**One other problem tesseract ocr** is that it **requires tesseract executable** so, i foresaw this issue, because this generally causes problem in deployment as we need to download it using linux,and i have faced duch problems before when trying to download chrome.exe on aws ec2(faced a lot of problem but ultimately downloaded it),
            moreover this time i was thinking of using pythonwnywhere which requires us to take manual permission to get any thirdparty app downloaded. Thankfully, its already installed in pyhtonanywhere env, so this was not a roadblock for me but many may face problems in such cases.         
            <br> Not used SpringBoot: I have done a lots and lots of coding in java and particularly in springboot framework, so my first choice for the restcontroller was spring boot and i integrated it too, but **problem came in deployment**,
            i planned to use elasticbeanstalk and after spending countless hours,i ended up with a pool of errors, therefore decided to reduce complexity and move on with python itsel and since we didnt had any extensive use of mongodb, i decided to integrate it with python itself, and it worked like a charm....

**problems**
         <br> -major problems came in deployment
        <br>-Earlier was using github pages for react deployment, but it didnt work out, later i found out i was using <Router> and<Routes> so removed them but it didnt work out, so had to move to firebase, on it deployment was quite simple
        <br>-used pythonanywhere, one major problem encountered was **using "pip" instead of "pip3.9"**, or whatever version required
          
            
            
            
  
