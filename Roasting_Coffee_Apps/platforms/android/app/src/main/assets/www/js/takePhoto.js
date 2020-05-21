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
        //konvert base64 string ke image
        document.getElementById('photo').src = "data:image/jpeg;base64," +imgURI;
        sessionStorage.setItem("Gambar", imgURI);
        console.log(imgURI);
        var op = document.getElementById('select-box1');
        var opsinilai =  op.options[op.selectedIndex].value;
        console.log(opsinilai);
        if(opsinilai!=null){
            if(opsinilai == "LeAdam"){
                $.ajax({
                    url: "http://192.168.1.6:5000//LeNet-Adam",
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
            } else if(opsinilai == "LeNadam"){
                $.ajax({
                    url: "http://192.168.1.6:5000//LeNet-Nadam",
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
            } else if(opsinilai=="LeSGD"){
                $.ajax({
                    url: "http://192.168.1.6:5000//LeNet-SGD",
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
            } else if(opsinilai == "AleAdam"){
                $.ajax({
                    url: "http://192.168.1.6:5000//AlexNet-Adam",
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
            } else if(opsinilai == "AleNadam"){
                $.ajax({
                    url: "http://192.168.1.6:5000//AlexNet-Nadam",
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
            } else if(opsinilai == "AleSGD"){
                $.ajax({
                    url: "http://192.168.1.6:5000//AlexNet-SGD",
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
            } else if(opsinilai == "MinAdam"){
                $.ajax({
                    url: "http://192.168.1.6:5000//MiniVgg-Adam",
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
            } else if(opsinilai == "MinNadam"){
                $.ajax({
                    url: "http://192.168.1.6:5000//MiniVgg-Nadam",
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
            } else if(opsinilai == "MinSGD"){
                $.ajax({
                    url: "http://192.168.1.6:5000//MiniVgg-SGD",
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
            }
        } else{
            alert("Pilih Arsitektur Dan Optimasi Yang Digunakan")
        }
    },

    //jika gagal
    wta: function(msg){
        document.getElementById('msg').textContent = "Fail to Access The Camera";
    }
};

//mengecek device ready / tidak
document.addEventListener('deviceready', app.init);


