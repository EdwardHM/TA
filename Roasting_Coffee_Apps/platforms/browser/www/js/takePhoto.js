let app = {
    init: function(){
        document.getElementById('btn').addEventListener('click', app.takephoto);
    },
    takephoto: function(){
        let opts = {
            quality: 100,
            destinationType: Camera.DestinationType.DATA_URL,
            sourceType: Camera.PictureSourceType.CAMERA,
            mediaType: Camera.MediaType.PICTURE,
            encodingType: Camera.EncodingType.JPEG,
            cameraDirection: 1,
            targetWidth: 400,
            targetHeight: 300
        };
        
        navigator.camera.getPicture(app.ftw, app.wta, opts);
    },

    //jika sukses
    ftw: function(imgURI){
        //  alert(imgURI);
        sessionStorage.setItem('image', imgURI);
        // document.getElementById('msg').textContent = imgURI;
        //konvert base64 string ke image
        document.getElementById('photo').src = "data:image/jpeg;base64," +imgURI;
        console.log(imgURI);
        // sessionStorage.setItem("img",imgURI);
        $.ajax({
            url: "http://192.168.1.7:5000//LeNet-Adam",
            type: "POST",
            datatype: "json",
            crossDomain: true,
            data: JSON.stringify({ imgBs64:imgURI }),
            cache: false,
            processData: false,

            success: function(result){
                console.log("berhasil predict");
                console.log(result)
                var hasil = result;
                document.getElementById('box-hasil').style.display = "block";
                document.getElementById('msg').innerHTML = hasil;
            }
        });
    },

    //jika gagal
    wta: function(msg){
        document.getElementById('msg').textContent = "Fail to Access The Camera";
    }
};

//mengecek device ready / tidak
document.addEventListener('deviceready', app.init);


