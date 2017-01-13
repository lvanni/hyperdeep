<?php 

// STATIC VALUES
define("CORPUS", "Campagne2017");

/**
 * @param unknown $api_url_request
 */
function api_call($api_url_request) {
	
	$request_id = hash("md5", $api_url_request);
	if (file_exists ("www/" . CORPUS . "/cache/" . $request_id)) {
		$result = json_decode(file_get_contents("www/" . CORPUS . "/cache/" . $request_id));
	} else {
		$result = json_decode(file_get_contents($api_url_request));
		while(!$result || strpos($result->status, WAITING_STAT) !== false) {
			sleep(0.5);
			$result = json_decode(file_get_contents($api_url_request));
		}
		file_put_contents("www/" . CORPUS . "/cache/" . $request_id, json_encode($result));
	}
	return $result;
}

// GET CORPUS META-DATAs
$metadata = api_call(API_URL . SELECT . "?" . P1 . "=" . CORPUS);
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
	$text = api_call(API_URL . READ . "?" . P1 . "=" . CORPUS . "&partition=discours:" . $last_discours);
	$text = $text->data;
} else {
	// LAST DISCOURS DATE
	$last_date = reset($date);
	
	// LAST TEXT
	$text = api_call(API_URL . READ . "?" . P1 . "=" . CORPUS . "&partition=date:" . $last_date);
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

// SPECIFICITE
$specificite_tmp = api_call(API_URL . SPEC . "?" . P1 . "=" . CORPUS . "&partition=discours:" . $discours);
$specificite_tmp = $specificite_tmp->data->FORME;
$specificite = array();

// THEME-CLOUD
$words_cloud = array();
$min_value = 0;
$max_value = 0;
foreach ($specificite_tmp as $s) {
	$z = floatval($s->z);
	if($min_value > $z) {
		$min_value = $z;
	}
	if ($max_value < $z) {
		$max_value = $z;
	}
}
$ratio = 50 / exp($max_value+abs($min_value));

$cpt=0;
foreach ($specificite_tmp as $s) {
	if ($cpt > 100) {
		break;
	}
	$z = floatval($s->z);
	$specificite[$s->word] = $z;
	$frequency = array();
	$frequency["text"] = $s->word;
	$frequency["size"] = exp($z+abs($min_value))*$ratio;
	//echo $s->word . ": " . $frequency["size"] . "<br />";
	array_push($words_cloud, $frequency);
	$cpt++;
}
//echo $cpt . " mots";
//print_r($discours);
//print_r($candidat);
//print_r($words_cloud);

?>