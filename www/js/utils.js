$(document).foundation();

$(".selected-item").hover(
  function() {
    $( this ).toggleClass("selected-menu");
});

$(".profil-candidat").hover(
  function() {
    $( this ).toggleClass("profil-selected");
});

$( document ).ready(function() {
    //console.log( "ready!" );
});