<?php 

// STATIC VALUES
define("CORPUS", "Campagne2017");

// GET CORPUS META-DATAs
$metadata = json_decode(file_get_contents(API_URL . SELECT . "?" . P1 . "=" . CORPUS));
while($metadata->status == WAITING_STAT) {
	sleep(0.5);
	$metadata = json_decode(file_get_contents(API_URL . SELECT . "?" . P1 . "=" . CORPUS));
}
$partitions = $metadata->data->__VAR__;
$date = $partitions->date;
$discours = $partitions->discours;
rsort($date);

if (isset($_GET["candidat"])) {
	$last_discours = ""; 
	
	// LAST DISCOURS
	foreach ($discours as $d) {
		if (strpos($d, $_GET["candidat"]) !== false && strcmp($last_discours, $d) < 0) {
			$last_discours = $d;
		}
	}
	
	// LAST TEXT
	$text = json_decode(file_get_contents(API_URL . READ . "?" . P1 . "=" . CORPUS . "&partition=discours:" . $last_discours));
	while($text->status == WAITING_STAT) {
		sleep(0.5);
		$text = json_decode(file_get_contents(API_URL . READ . "?" . P1 . "=" . CORPUS . "&partition=discours:" . $last_discours));
	}
	$text = $text->data;
} else {
	// LAST DISCOURS DATE
	$last_date = reset($date);
	
	// LAST TEXT
	$text = json_decode(file_get_contents(API_URL . READ . "?" . P1 . "=" . CORPUS . "&partition=date:" . $last_date));
	while($text->status == WAITING_STAT) {
		sleep(0.5);
		$text = json_decode(file_get_contents(API_URL . READ . "?" . P1 . "=" . CORPUS . "&partition=date:" . $last_date));
	}
	$text = $text->data;
}

// LAST DISCOURS ID
$first_line = array_shift ($text);
$discours = explode(" ", explode("discours_", $first_line)[1])[0];
$candidat = explode(" ", explode("candidat_", $first_line)[1])[0];
$date = explode(" ", explode("date_", $first_line)[1])[0];
$pretty_print_date = "";
$mois = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"];
$date_args = explode("-", $date); 
$pretty_print_date = intval($date_args[2]) . " " . $mois[intval($date_args[1])-1] . " " . $date_args[0];
$type = explode(" ", explode("type_", $first_line)[1])[0];
$source = explode(" ", explode("source_", $first_line)[1])[0];

//print_r($discours);
//print_r($candidat);

?>