<!doctype html>
<html lang="en">
<%@ page import="java.sql.*"%>
<%@ page import="java.io.BufferedReader" %>
<%@ page import="java.io.FileReader" %>
<%@ page import="java.io.File" %>
<head>
    <!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>JLU · Bird</title>
    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="./assets/vendor/bootstrap/css/bootstrap.min.css">
    <link href="./assets/vendor/fonts/circular-std/style.css" rel="stylesheet">
    <link rel="stylesheet" href="./assets/libs/css/style.css">
    <link rel="stylesheet" href="./assets/vendor/fonts/fontawesome/css/fontawesome-all.css">
    <link rel="stylesheet" type="text/css" href="./assets/vendor/datatables/css/dataTables.bootstrap4.css">
    <link rel="stylesheet" type="text/css" href="./assets/vendor/datatables/css/buttons.bootstrap4.css">
    <link rel="stylesheet" type="text/css" href="./assets/vendor/datatables/css/select.bootstrap4.css">
    <link rel="stylesheet" type="text/css" href="./assets/vendor/datatables/css/fixedHeader.bootstrap4.css">


	<!-- activity图 -->
    <link rel="stylesheet" href="../assets/vendor/charts/chartist-bundle/chartist.css">
</head>



<body>
    <!-- ============================================================== -->
    <!-- main wrapper -->
    <!-- ============================================================== -->
    <div class="dashboard-main-wrapper">
        <!-- ============================================================== -->
        <!-- navbar -->
        <!-- ============================================================== -->
        <div class="dashboard-header">
            <nav class="navbar navbar-expand-lg bg-white fixed-top">
                <a class="navbar-brand" href=#>JLU · Bird</a>
            </nav>
        </div>
        <!-- ============================================================== -->
        <!-- end navbar -->
        <!-- ============================================================== -->
        <!-- ============================================================== -->
        <!-- left sidebar -->
        <!-- ============================================================== -->
        <div class="nav-left-sidebar sidebar-dark">
            <div class="menu-list">
                <nav class="navbar navbar-expand-lg navbar-light">
                    <a class="d-xl-none d-lg-none" href="#">Dashboard</a>
                    <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                        <span class="navbar-toggler-icon"></span>
                    </button>
                    <div class="collapse navbar-collapse" id="navbarNav">
                        <ul class="navbar-nav flex-column">
                            <li class="nav-divider">
                                MENU
                            </li>
                            <li class="nav-item ">
                                <a class="nav-link" href="#" data-target="#submenu-1" aria-controls="submenu-1"> About the Project</a>
                            </li>
                            <li class="nav-item ">
                                <a class="nav-link active" href="#" data-target="#submenu-2" aria-controls="submenu-2">Birdcall Identification</a>
                            </li>
                            <li class="nav-item ">
                                <a class="nav-link" href="#" data-target="#submenu-3" aria-controls="submenu-3">Background</a>
                            </li>
                            <li class="nav-item ">
                                <a class="nav-link" href="#" data-target="#submenu-4" aria-controls="submenu-4">Workflow</a>
                            </li>
                            <li class="nav-item ">
                                <a class="nav-link" href="#" data-target="#submenu-5" aria-controls="submenu-5">Visualization</a>
                            </li>
                            <li class="nav-item ">
                                <a class="nav-link" href="#" data-target="#submenu-6" aria-controls="submenu-6">Contact Me</a>
                            </li>
                        </ul>
                    </div>
                </nav>
            </div>
        </div>

        <!-- ============================================================== -->
        <!-- end left sidebar -->
        <!-- ============================================================== -->
        <!-- ============================================================== -->
        <!-- wrapper  -->
        <!-- ============================================================== -->
        <div class="dashboard-wrapper">
            <div class="container-fluid  dashboard-content">

                <div class="page-section" id="overview">
                    <!-- ============================================================== -->
                    <!-- overview  -->
                    <!-- ============================================================== -->
                    <div class="row">
                        <div class="col-xl-12 col-lg-12 col-md-12 col-sm-12 col-12">
                            <h2>Birdcall Identification</h2>
                            <p class="lead">Welcome to JLU Bird! You could upload your audio, and identify bird species with our interfaces.
							Please input complete and accurate information, as it will improve the accuracy of birdcall identification and provide helpful insights for data analysis and algorithm updates.</p>
                            <ul class="list-unstyled arrow">
                                <li>The system can identify 397 bird species from 4 locations in North America.</li>
                                <li>The system supports .wav and .ogg files.</li>
                                <li>The more complete the information you provide, the more accurate the Identification results.</li>
                                <li>The identification process takes a relatively long time, please wait.</li>
                            </ul>
                        </div>
                    </div>
                    <!-- ============================================================== -->
                    <!-- end overview  -->
                    <!-- ============================================================== -->
                </div>


							<div>
							<%
							out.println("<audio id=\"player\" src=\"playaudio.ogg\" controls=\"controls\" style=\"width: 100%\">Your browser does not support the audio element.</audio>");
							%>
							<!--<audio id="player" src="playaudio.ogg" controls="controls" style="width: 100%;background:#F1F3F4;">
								Your browser does not support the audio element.
							</audio>-->


							
							</div>



                <div class="row">
                    <!-- ============================================================== -->
                    <!-- data table  -->
                    <!-- ============================================================== -->
                    <div class="col-xl-12 col-lg-12 col-md-12 col-sm-12 col-12">
                        <div class="card">
                            <div class="card-header">
                                <h5 class="mb-0">Results</h5>
                                <p>The format of time stamp is "h: m: s". The last audio slice is extended to 5 seconds. The identification results are ranked according to the occurrence possibility.</p>
                            </div>

                            <div class="card-body">
								



                                <div class="table-responsive">
                                    <table id="example" class="table table-striped table-bordered second" style="width:100%">
                                        <thead>
                                            <tr>
												<th>Audio&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</th>
                                                <th>Begin</th>
                                                <th>End&nbsp;&nbsp;</th>
                                                <th>Bird 1</th>
                                                <th>Bird 2</th>
                                                <th>Bird 3</th>
                                            </tr>
                                        </thead>


										<tbody>
										<%
										    String web_path = "D:/tomcat/webapps/ROOT/";
											request. setCharacterEncoding("UTF-8");
											BufferedReader buf = null ;		// 声明对象
                                            buf = new BufferedReader(new FileReader(new File(web_path+"ctable.txt"))) ;	// 将字节流变为字符流
                                            try{
                                                String str = null;
                                                int i = 0;
                                                while((str = buf.readLine()) != null)
                                                {
                                                        String[] str_split = str.split("@");
                                                        String f = str_split[0];
														String t = str_split[1];
														String b1 = str_split[2];
														String b2 = str_split[3];
														String b3 = str_split[4];

                                                        out.println("<tr>");
														out.println("<td>");
														String tempstr = i+"";
														out.println("<audio id=\"player\" src=\"playaudios\\playaudio-" + tempstr + ".ogg\" controls=\"controls\" style=\"width: 100%\">Your browser does not support the audio element.</audio>");
														i++;
														out.println("</td>");
														out.println("<td>");
														out.println(f);
														out.println("</td>");
														out.println("<td>");
														out.println(t);
														out.println("</td>");
														out.println("<td>");
														if (b1.equals("nocall"))
															out.println("   ");
														else
															{out.println("<a href='https://ebird.org/species/" + b1 + "' target='_blank' >");
															out.println(b1);
															out.println("</a>");}

														out.println("</td>");
														out.println("<td>");
														if (b2.equals("nocall"))
															out.println("   ");
														else
															{out.println("<a href='https://ebird.org/species/" + b2 + "' target='_blank' >");
															out.println(b2);
															out.println("</a>");}
														out.println("</td>");
														out.println("<td>");
														if (b3.equals("nocall"))
															out.println("   ");
														else
															{out.println("<a href='https://ebird.org/species/" + b3 + "' target='_blank' >");
															out.println(b3);
															out.println("</a>");}
														out.println("</td>");
														out.println("</tr>");
                                            }
                                            }catch(Exception e){
                                                e.printStackTrace() ;	// 输出信息
                                            }

										%>
										</tbody>


                                        <tfoot>
                                            <tr>
												<th>Audio&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</th>
                                                <th>Begin</th>
                                                <th>End&nbsp;&nbsp;</th>
                                                <th>Bird 1</th>
                                                <th>Bird 2</th>
                                                <th>Bird 3</th>
                                            </tr>
                                        </tfoot>
                                    </table>
                                </div>

                            </div>
							<script language="JavaScript">  
								function   goback()   
								  {  
								  //
										history.go(-1);
								  }     
							 </script>
							 

							<!--<div class="form-group">-->
								<div class="col-sm-12 pl-0">
									<p class="text-right">
										<button name="1" class="btn btn-space btn-primary" onclick="javascript:location.reload();">Refresh</button>
										<button name="2" class="btn btn-space btn-secondary" onclick="goback();">&nbsp;&nbsp;Back&nbsp;&nbsp;</button>
									</p>
                                </div>
								</br>
                            <!--</div>-->


                        </div>
                    </div>
                    <!-- ============================================================== -->
                    <!-- end data table  -->
                    <!-- ============================================================== -->
                </div>



				<div class="row">
                    <!-- ============================================================== -->
                    <!-- pie chart  -->
                    <!-- ============================================================== -->
                    <div class="col-xl-6 col-lg-6 col-md-6 col-sm-12 col-12">
                        <div class="card">
                            <h5 class="card-header">Active Species</h5>
                            <div class="card-body">

                                <canvas id="chartjs_pie"></canvas>

                            </div>
                        </div>
                    </div>
                    <!-- ============================================================== -->
                    <!-- end pie chart  -->
                    <!-- ============================================================== -->
<!-- ============================================================== -->
                    <!-- line chart with area  -->
                    <!-- ============================================================== -->
                    <div class="col-xl-6 col-lg-6 col-md-6 col-sm-12 col-12">
                        <div class="card">
                            <h5 class="card-header">Sound Events</h5>
                            <div class="card-body">
                                <div class="ct-chart-polar ct-golden-section"></div>
                            </div>
                        </div>
                    </div>
                    <!-- ============================================================== -->
                    <!-- end line chart with area  -->
                    <!-- ============================================================== -->
				</div>

            </div>
            <!-- ============================================================== -->
            <!-- footer -->
            <!-- ============================================================== -->
            <div class="footer">
                <div class="container-fluid">
                    <div class="row">
                        <div class="col-xl-6 col-lg-6 col-md-12 col-sm-12 col-12">
                            JLU Bird Call Identifier by Yichuan. Dashboard by <a href="https://colorlib.com/wp/">Colorlib</a>.
                        </div>
                        <div class="col-xl-6 col-lg-6 col-md-12 col-sm-12 col-12">

                        </div>
                    </div>
                </div>
            </div>
            <!-- ============================================================== -->
            <!-- end footer -->
            <!-- ============================================================== -->
        </div>
    </div>
    <!-- ============================================================== -->
    <!-- end main wrapper -->
    <!-- ============================================================== -->
    <!-- Optional JavaScript for Table-->
    <script src="./assets/vendor/jquery/jquery-3.3.1.min.js"></script>
    <script src="./assets/vendor/bootstrap/js/bootstrap.bundle.js"></script>
    <script src="./assets/vendor/slimscroll/jquery.slimscroll.js"></script>
    <script src="./assets/vendor/multi-select/js/jquery.multi-select.js"></script>
    <script src="./assets/libs/js/main-js.js"></script>
    <script src="https://cdn.datatables.net/1.10.19/js/jquery.dataTables.min.js"></script>
    <script src="./assets/vendor/datatables/js/dataTables.bootstrap4.min.js"></script>
    <script src="https://cdn.datatables.net/buttons/1.5.2/js/dataTables.buttons.min.js"></script>
    <script src="./assets/vendor/datatables/js/buttons.bootstrap4.min.js"></script>
    <script src="./assets/vendor/datatables/js/data-table.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.1.3/jszip.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/pdfmake/0.1.36/pdfmake.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/pdfmake/0.1.36/vfs_fonts.js"></script>
    <script src="https://cdn.datatables.net/buttons/1.5.2/js/buttons.html5.min.js"></script>
    <script src="https://cdn.datatables.net/buttons/1.5.2/js/buttons.print.min.js"></script>
    <script src="https://cdn.datatables.net/buttons/1.5.2/js/buttons.colVis.min.js"></script>
    <script src="https://cdn.datatables.net/rowgroup/1.0.4/js/dataTables.rowGroup.min.js"></script>
    <script src="https://cdn.datatables.net/select/1.2.7/js/dataTables.select.min.js"></script>
    <script src="https://cdn.datatables.net/fixedheader/3.1.5/js/dataTables.fixedHeader.min.js"></script>

    <!-- Optional JavaScript for Form Validation-->
    <script src="./assets/vendor/jquery/jquery-3.3.1.min.js"></script>
    <script src="./assets/vendor/bootstrap/js/bootstrap.bundle.js"></script>
    <script src="./assets/vendor/slimscroll/jquery.slimscroll.js"></script>
    <script src="./assets/vendor/parsley/parsley.js"></script>
    <script src="./assets/libs/js/main-js.js"></script>
    <script>
        $('#form').parsley();
    </script>
    <script>
        // Example starter JavaScript for disabling form submissions if there are invalid fields
        (function () {
            'use strict';
            window.addEventListener('load', function () {
                // Fetch all the forms we want to apply custom Bootstrap validation styles to
                var forms = document.getElementsByClassName('needs-validation');
                // Loop over them and prevent submission
                var validation = Array.prototype.filter.call(forms, function (form) {
                    form.addEventListener('submit', function (event) {
                        if (form.checkValidity() === false) {
                            event.preventDefault();
                            event.stopPropagation();
                        }
                        form.classList.add('was-validated');
                    }, false);
                });
            }, false);
        })();
    </script>

    <!-- Optional JavaScript for Form Elements -->
    <script src="./assets/vendor/jquery/jquery-3.3.1.min.js"></script>
    <script src="./assets/vendor/bootstrap/js/bootstrap.bundle.js"></script>
    <script src="./assets/vendor/slimscroll/jquery.slimscroll.js"></script>
    <script src="./assets/libs/js/main-js.js"></script>
    <script src="./assets/vendor/inputmask/js/jquery.inputmask.bundle.js"></script>
    <script>
        $(function (e) {
            "use strict";
            $(".date-inputmask").inputmask("dd/mm/yyyy"),
                $(".phone-inputmask").inputmask("(999) 999-9999"),
                $(".international-inputmask").inputmask("+9(999)999-9999"),
                $(".xphone-inputmask").inputmask("(999) 999-9999 / x999999"),
                $(".purchase-inputmask").inputmask("aaaa 9999-****"),
                $(".cc-inputmask").inputmask("9999 9999 9999 9999"),
                $(".ssn-inputmask").inputmask("999-99-9999"),
                $(".isbn-inputmask").inputmask("999-99-999-9999-9"),
                $(".currency-inputmask").inputmask("$9999"),
                $(".percentage-inputmask").inputmask("99%"),
                $(".decimal-inputmask").inputmask({
                    alias: "decimal",
                    radixPoint: "."
                }),

                $(".email-inputmask").inputmask({
                    mask: "*{1,20}[.*{1,20}][.*{1,20}][.*{1,20}]@*{1,20}[*{2,6}][*{1,2}].*{1,}[.*{2,6}][.*{1,2}]",
                    greedy: !1,
                    onBeforePaste: function (n, a) {
                        return (e = e.toLowerCase()).replace("mailto:", "")
                    },
                    definitions: {
                        "*": {
                            validator: "[0-9A-Za-z!#$%&'*+/=?^_`{|}~/-]",
                            cardinality: 1,
                            casing: "lower"
                        }
                    }
                })
        });
    </script>

    <!-- Optional JavaScript for Date Picker -->
    <script src="./assets/vendor/jquery/jquery-3.3.1.min.js"></script>
    <script src="./assets/vendor/bootstrap/js/bootstrap.bundle.js"></script>
    <script src="./assets/vendor/slimscroll/jquery.slimscroll.js"></script>
    <script src="./assets/libs/js/main-js.js"></script>
    <script src="./assets/vendor/datepicker/moment.js"></script>
    <script src="./assets/vendor/datepicker/tempusdominus-bootstrap-4.js"></script>
    <script src="./assets/vendor/datepicker/datepicker.js"></script>

	    <!-- Optional JavaScript for Pie Chart -->
    <!--script src="../assets/vendor/jquery/jquery-3.3.1.min.js"></script-->
    <script src="./assets/vendor/bootstrap/js/bootstrap.bundle.js"></script>
    <script src="./assets/vendor/slimscroll/jquery.slimscroll.js"></script>
    <script src="./assets/vendor/charts/charts-bundle/Chart.bundle.js"></script>
    <!--script src="../assets/vendor/charts/charts-bundle/chartjs.js"></script>
    <script src="../assets/libs/js/main-js.js"></script-->

	<script>            
(function(window, document, $, undefined) {
        "use strict";
        $(function() {

            if ($('#chartjs_pie').length) {
                var ctx = document.getElementById("chartjs_pie").getContext('2d');
                var myChart = new Chart(ctx, {
                    type: 'pie',
                    data: {

                        
						<%
						out.println("labels: [");
						BufferedReader buf1 = null ;		// 声明对象
                        buf1 = new BufferedReader(new FileReader(new File(web_path+"bcount.txt"))) ;	// 将字节流变为字符流
                        try{
                            String str1 = null;
                            while((str1 = buf1.readLine()) != null)
                            {
                                String[] str_split1 = str1.split("@");
                                out.println("\""+str_split1[0]+"\",");
                            }
                        }
                        catch(Exception e){
                            e.printStackTrace() ;	// 输出信息
                        }
						out.println("],");
						%>
                        datasets: [{
                            backgroundColor: [
                               "#5969ff",
                                "#ff407b",
                                "#25d5f2",
                                "#ffc750",
                                "#2ec551",
                                "#7040fa",
                                "#ff004e",
								"#ff99ee",
								"#00ffff",
								"#ff6161",
								"#70c24a",
								"#ff0000",
								"#ff6600",
								"#0088cc",
                            ],
                            
													<%
													//data: [12, 19,]

						//labels: ["M", "T",],
						out.println("data: [");
						BufferedReader buf2 = null ;		// 声明对象
                        buf2 = new BufferedReader(new FileReader(new File(web_path+"bcount.txt"))) ;	// 将字节流变为字符流
                        try{
                            String str2 = null;
                            while((str2 = buf2.readLine()) != null)
                            {
                                String[] str_split2 = str2.split("@");
                                out.println(str_split2[1]+",");
                            }
                        }
                        catch(Exception e){
                            e.printStackTrace() ;	// 输出信息
                        }

						out.println("]");
						%>
                        }]
                    },
                    options: {
                           legend: {
                        display: true,
                        position: 'bottom',

                        labels: {
                            fontColor: '#71748d',
                            fontFamily: 'Circular Std Book',
                            fontSize: 14,
                        }
                    },

                    
                }
                });
            }

        });

})(window, document, window.jQuery);
	</script>

	    <!-- Optional JavaScript for bird activity -->
    <script src="./assets/vendor/jquery/jquery-3.3.1.min.js"></script>
    <script src="./assets/vendor/bootstrap/js/bootstrap.bundle.js"></script>
    <script src="./assets/vendor/slimscroll/jquery.slimscroll.js"></script>
    <script src="./assets/vendor/charts/chartist-bundle/chartist.min.js"></script>
    <!--script src="../assets/vendor/charts/chartist-bundle/Chartistjs.js"></script-->
    <script src="./assets/libs/js/main-js.js"></script>
	<script>
	(function(window, document, $, undefined) {
    "use strict";
    $(function() {



		        if ($('.ct-chart-polar').length) {
            new Chartist.Line('.ct-chart-polar', {
                labels: [1, 2, 3, 4, 5, 6, 7, 8],
															<%
															//                    series: [
                      //  [1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
                   // ]
                                            request. setCharacterEncoding("UTF-8");
											BufferedReader buf3 = null ;		// 声明对象
                                            buf3 = new BufferedReader(new FileReader(new File(web_path+"ctable.txt"))) ;	// 将字节流变为字符流
											out.println("series: [");
											out.println("[");
											request. setCharacterEncoding("UTF-8");
											try{
                                                String str = null;
                                                int i = 0;
                                                while((str = buf3.readLine()) != null)
                                                {
                                                        String[] str_split = str.split("@");
														String b1 = str_split[2];
														String b2 = str_split[3];
														String b3 = str_split[4];

														if (b3.equals("nocall"))
														{
															if (b2.equals("nocall"))
															{
																if (b1.equals("nocall"))
																{
																	out.println("0, ");
																}
																else
																{
																	out.println("1, ");
																}
															}
															else
															{
																out.println("2, ");
															}
														}
														else
														{
															out.println("3, ");
														}
                                            }
                                            }catch(Exception e){
                                                e.printStackTrace() ;	// 输出信息
                                            }

											out.println("]");
											out.println("]");
										%>
            }, {
                high: 3,
                low: 0,
                showArea: true,
                showLine: false,
                showPoint: false,
                fullWidth: true,
                axisX: {
                    showLabel: false,
                    showGrid: false
                },
               
            });
        }








    });

})(window, document, window.jQuery);
	</script>



</body>

</html>