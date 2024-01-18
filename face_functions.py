def append_face_to_file():
    print(analyticsDict)
    confidence = 0
    nervous = 0
    neutral = 0

    for key in analyticsDict:
        if key == "happy":
            confidence += analyticsDict[key]
        if key == "sad":
            nervous += analyticsDict[key]
        if key == "angry":
            nervous += analyticsDict[key]
        if key == "neutral":
            neutral += analyticsDict[key]
        if key == "disgust":
            nervous += analyticsDict[key]
        if key == "fear":
            nervous += analyticsDict[key]
        if key == "surprise":
            nervous += analyticsDict[key]
    
    total = confidence + nervous + neutral
    if(total != 0):
        confidence = ((confidence + neutral/2)/total) * 100
        neutral = (neutral/total) * 100
        nervous = (nervous /total) * 100

    mainstr = "confidence: " + str(round(confidence, 2)) + "%\t nervousness: " + str(round(nervous, 2)) + "%\t neutral: " + str(round(neutral, 2)) + "%\n"
    print(mainstr)  
    print(analyticsDict)

    file1 = open("report_file.txt", "a")
    file1.write(mainstr)
    file1.write("\n")
    file1.close()

    for key in analyticsDict:
        analyticsDict[key] = 0
    print(analyticsDict)
    
    return "done"

def update_face_analytics(face_properties):
    if (len(face_properties)!=0 ):
            analyticsDict[face_properties[0]['label']] +=1
        

analyticsDict = {
  "angry": 0,
  "disgust": 0,
  "fear": 0,
  "happy": 0,
  "sad": 0,
  "surprise": 0,
  "neutral": 0
}