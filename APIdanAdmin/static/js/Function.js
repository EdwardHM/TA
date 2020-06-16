function Ambildata(){
    $('.AmbilData').click(function (event) {
        var $row = $(this).parents('tr');
        var desc = $row.find('input[name="kelas"]').val();
        var nama = $row.find('input[name="name"]').text;
    
        alert('description: ' + desc + 'nama '+nama);
    });
}
