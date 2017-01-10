<?php 
// DISPLAY ERROR
if ($_SERVER['SERVER_ADDR'] == "::1") { // localhost
	error_reporting(E_ALL);
	ini_set('display_errors', '1');
}

// unlimited memory usage => !!!!!
ini_set('memory_limit', '-1');

// START PHP SESSION
session_start();

// API CALL
define("API_URL", "http://localhost/hyperbase/api/");

// RESULTS VALUES
define("WAITING_STAT", "__WAIT__");

// PARAMETERS
define("P1", "corpus_id");
define("P2", "partition");

// FUNCTIONS
define("SELECT", "select.php");
define("READ", "read.php");

?>
