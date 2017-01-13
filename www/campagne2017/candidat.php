<div class="top-bar menu-candidat ">
    <ul class="dropdown menu expanded medium-text-center" >
      <?php foreach ($partitions->candidat as $candidat_i) { ?>
      	  <?php if (!strcasecmp ($candidat_i, $candidat)) { ?>
	      	<li><a href="#"><img class="profil-selected" alt="valls" src="www/img/<?php echo $candidat_i ?>.png" /></a></li>
	      <?php } else { ?>
	      	<li><a href="?candidat=<?php echo $candidat_i ?>"><img class="profil-candidat" alt="valls" src="www/img/<?php echo $candidat_i ?>.png" /></a></li>
	  	  <?php } ?>    
      <?php } ?>
    </ul>
</div>