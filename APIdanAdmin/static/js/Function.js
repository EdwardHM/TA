function  getId(element) {
;
    // var kolom = element.parentNode.cellIndex;
    var baris = element.parentNode.parentNode.rowIndex;
    var nama = document.getElementById("ListFoto").rows[baris].cells.item(3).innerHTML;
    var kelas = document.getElementById("ListFoto").rows[baris].cells[1].childNodes[1].value;

    $.ajax({
        url: "http://192.168.1.7:5001/Move",
        type: "POST",
        datatype: "json",
        crossDomain: true,
        data:JSON.stringify( { gambar:nama, kls:kelas} ),
        cache: false,
        processData: false,

        success: function(result){
            console.log("berhasil pindahkan gambar ke Dataset");
            alert(result)
            // href ke route yang di python
            window.location.href = "/";
        }
    });
}

function IdHapus(element){
    var baris = element.parentNode.parentNode.rowIndex;
    var nama = document.getElementById("ListFoto").rows[baris].cells.item(3).innerHTML;
    $.ajax({
        url: "http://192.168.1.7:5001/Delete",
        type: "POST",
        datatype: "json",
        crossDomain: true,
        data:JSON.stringify( { gambar:nama } ),
        cache: false,
        processData: false,

        success: function(result){
            console.log("berhasil pindahkan gambar ke Dataset");
            alert(result)
            // href ke route yang di python
            window.location.href = "/";
        }
    });
}