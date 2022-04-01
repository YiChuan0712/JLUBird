<!doctype html>
<html lang="en">
<%@ page import="java.sql.*"%>
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
                            <p class="lead">Welcome to JLU Bird! You could upload your soundscape audio with its information, and identify bird species with our interfaces. 
							Please provide complete and accurate audio information, as it will improve the accuracy of birdcall identification and provide helpful information for data analysis and algorithm study.</p>
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
							<audio id="player" src="playaudio.ogg" controls="controls" style="width: 100%;background:#F1F3F4;">
								Your browser does not support the audio element.
							</audio>


							
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
										
											request. setCharacterEncoding("UTF-8");
											try {
												Class.forName("com.mysql.jdbc.Driver");
												String url = "jdbc:mysql://localhost:3306/bird?serverTimezone=UTC";
												String usernam = "root";  //数据库用户名
												String password = "root";  //数据库用户密码
												Connection conn = DriverManager.getConnection(url, usernam, password);  //连接状态

												if (conn != null) 
												{
													//usernam = request.getParameter("username");
													//password = request.getParameter("password");
													PreparedStatement pStmt = conn.prepareStatement("select * from ctable order by f");
													ResultSet rs = pStmt.executeQuery();
													int i = 0;
													while(rs.next())
													{
														String f = rs.getObject("f").toString();
														String t = rs.getObject("t").toString();
														String b1 = rs.getObject("b1").toString();
														String b2 = rs.getObject("b2").toString();
														String b3 = rs.getObject("b3").toString();
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

												}
												else{
													out.print("连接失败！");
												}
											}catch (ClassNotFoundException e) {
												e.printStackTrace();
												out.print("1");
											}catch (SQLException e){
												e.printStackTrace();
												out.print("");
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

</body>

</html>