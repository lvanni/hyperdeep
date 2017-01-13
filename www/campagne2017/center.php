<h4 style='color:grey;'>Dernier discours enregistré :</h4>
<h5><?php echo $candidat ?>, le <?php echo $pretty_print_date ?> -  <?php echo $type ?> - <?php echo $source ?></h5>

<div>
<?php 
$cpt = 0;
foreach ($text as $line) {
	if ($cpt == 10) {
		echo "</div>";
		echo '<a href="" onclick="$(\'#text-part2\').fadeIn(); $(this).hide();">Charger tour le texte<br /><br /></a>';
		echo "<div id='text-part2' style='display:none;'>";
	}
	$cpt++;
	$words = explode(" ", $line);
	foreach ($words as $word) {
		if (isset($specificite[$word]) && $specificite[$word] > 2) {
			echo '<span class="specificite" data-tooltip aria-haspopup="true" class="has-tip"  title="spécificité: +' . strval($specificite[$word]) .'">' . $word . '</span> ';
		} else {
			echo $word . " ";
		}
	}
} 
?>

</div>