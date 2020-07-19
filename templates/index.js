function extract_text(){
    document.getElementById("demo").innerHTML = "Hello World!"
    console.log(1);
    // var xhr = new XMLHttpRequest();
    // var url = "/extract" + encodeURIComponent(JSON.stringify({"text": "hey@mail.com", "sentiment": "101010"}));
    // xhr.open("GET", url, true);
    // xhr.setRequestHeader("Content-Type", "application/json");
    // xhr.onreadystatechange = function () {
    //     if (xhr.readyState === 4 && xhr.status === 200) {
    //         var json = JSON.parse(xhr.responseText);
    //         console.log(json.email + ", " + json.password);
    //     }
    // };
    // xhr.send();
}