<div id="footer1">
	blabalbal
</div>
<div id="footer2">
		<br />
		<a href="#">Hyperbase Web Edition</a> - 
		<a href="http://logometrie.unice.fr/" target="blank">Logométrie</a> - 
		<a href="<?php echo HTML_ROOT_PATH ?>ui/common/mentions_legales.html" target="blank">Mentions Légales</a> - 
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