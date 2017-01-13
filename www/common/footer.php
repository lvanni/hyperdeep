<div id="footer1" class="row expanded">
	<div class="small-12 large-4 columns" style="border-right: thin lightgrey solid;">
	<i class="fi-flag footer-icon"></i>
	<h5>Mesure Du Discours</h5>
	<ul>
	  <li>Projet de recherche <a href="http://univ-cotedazur.fr/english/idex-uca-jedi/ucajedi-project" target="blank">IDEX (JEDI)</a></li>
	  <li>Laboratoire UMR7320 : <a href="http://bcl.cnrs.fr" target="blank">Bases, Corpus, Langage</a></li>
	  <li>Équipe <a href="http://logometrie.unice.fr" target="blank">Logométrie</a></li>
	</ul>
  	</div>
  	
	<div class="small-12 large-4 columns" style="border-right: thin lightgrey solid;">
	<i class="fi-info footer-icon"></i>
	<h5>Plus d'informations</h5>
	<ul>
	  <li><a href="http://lexicometrica.univ-paris3.fr/jadt/jadt2016/01-ACTES/86038/86038.pdf" target="blank">Deep Learning et Discours politique</a></li>
	  <li>ADT - Méthodes statistiques : <a href="https://fr.wikipedia.org/wiki/Hyperbase" target="blank">Hyperbase</a></li>
	  <li>Plateforme expérimentale et API : <a href="http://hyperbase.unice.fr" target="blank">Hyperbase Web</a></li>
	</ul>
	</div>
    
    <div class="small-12 large-4 columns">
    <i class="fi-mail footer-icon"></i>
    <h5>Contact</h5>
    <ul>
	  <li>Responsables contenus & analyses : <a href="mailto:damonDOTmayaffreATuniceDOTfr" target="blank">Damon Mayaffre</a>, <a href="mailto:damonDOTmayaffreATuniceDOTfr" target="blank">Camille Bouzereau</a></li>
	  <li>Responsables scientifiques : <a href="mailto:laurentDOTvanniATuniceDOTfr" target="blank">Laurent Vanni</a>, <a href="mailto:damonDOTmayaffreATuniceDOTfr" target="blank">Frédérique Précioso</a>, <a href="mailto:damonDOTmayaffreATuniceDOTfr" target="blank">Mélanie Ducoffe</a></li>
	  <li>Responsable du design du site & du développement informatiques : <a href="mailto:laurentDOTvanniATuniceDOTfr" target="blank">Laurent Vanni</a></li>
	</ul>
    </div>
</div>
<div id="footer2">
		<br />
		<a href="http://mesure-du-discours.unice.fr">MESURE DU DISCOURS</a> - 
		<a href="http://logometrie.unice.fr/" target="blank">Logométrie</a> - 
		<a href="mentions_legales.html" target="blank">Mentions Légales</a> - 
		<a href="http://www.unice.fr/bcl/" target="blank">UMR 7320 : Bases, Corpus, Langage</a> - 
		<a href="mailto:laurent.vanni@unice.fr">Contact</a>
</div>


	<script src="www/lib/foundation-6.2.4-complete/js/vendor/jquery.js"></script>
    <script src="www/lib/foundation-6.2.4-complete/js/vendor/what-input.js"></script>
    <script src="www/lib/foundation-6.2.4-complete/js/vendor/foundation.min.js"></script>
	<script src="www/lib/d3/d3.min.js"></script>
	<script src="www/lib/nvd3/nv.d3.js"></script>
	<script src="www/lib/d3/d3.layout.cloud.js"></script>
    
    <script src="www/js/utils.js?<?php echo uniqid(); ?>"></script>
    
	<!-- On document ready -->
	<script>
		$( document ).ready(function() {
			$(document).foundation();
		});
	</script>
	
	
	<script>
	var words_cloud = <?php echo json_encode($words_cloud) ?>;
	console.log($("#cloud-container").width());
	var fill = d3.scale.category20();
	  d3.layout.cloud().size([$("#cloud-container").width(), 300])
	      .words(words_cloud)
	      .padding(5)
	      //.rotate(function() { return ~~(Math.random() * 2) * 90; })
	      .font("Impact")
	      .fontSize(function(d) { return d.size; })
	      .on("end", draw)
	      .start();
	  function draw(words) {
	    d3.select("#cloud-container").append("svg")
	        .attr("width", $("#cloud-container").width())
	        .attr("height", 300)
	      .append("g")
	        .attr("transform", "translate(" + $('#cloud-container').width()/2 + ",150)")
	      .selectAll("text")
	        .data(words)
	      .enter().append("text")
	        .style("font-size", function(d) { return d.size + "px"; })
	        .style("font-family", "Impact")
	        .style("fill", function(d, i) { return fill(i); })
	        .attr("text-anchor", "middle")
	        .attr("transform", function(d) {
	          return "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")";
	        })
	        .text(function(d) { return d.text; });
	  }
	 </script>
	

</body>
</html>