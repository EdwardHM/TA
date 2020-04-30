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
            targetWidth: 300,
            targetHeight: 400
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
        sessionStorage.setItem("img",imgURI);
    },

    //jika gagal
    wta: function(msg){
        document.getElementById('msg').textContent = "Fail to Access The Camera";
    }
};

//mengecek device ready / tidak
document.addEventListener('deviceready', app.init);


