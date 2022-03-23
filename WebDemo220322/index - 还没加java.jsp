<!doctype html>
<html lang="en">

<head>
    <!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>吉大 · 听鸟</title>
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
                                MENU 目录
                            </li>
                            <li class="nav-item ">
                                <a class="nav-link" href="#" data-target="#submenu-1" aria-controls="submenu-1"> About the Project 项目简介</a>
                            </li>
                            <li class="nav-item ">
                                <a class="nav-link active" href="#" data-target="#submenu-2" aria-controls="submenu-2">Bird Call Identification 鸟声识别</a>
                            </li>
                            <li class="nav-item ">
                                <a class="nav-link" href="#" data-target="#submenu-3" aria-controls="submenu-3">Background 研究背景</a>
                            </li>
                            <li class="nav-item ">
                                <a class="nav-link" href="#" data-target="#submenu-4" aria-controls="submenu-4">Workflow 实现方案</a>
                            </li>
                            <li class="nav-item ">
                                <a class="nav-link" href="#" data-target="#submenu-5" aria-controls="submenu-5">Visualization 可视化</a>
                            </li>
                            <li class="nav-item ">
                                <a class="nav-link" href="#" data-target="#submenu-6" aria-controls="submenu-6">Contact Me 联系我</a>
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
                            <h2>Bird Call Identification 鸟声识别</h2>
                            <p class="lead">欢迎体验鸟声识别功能！您可以填写音频背景信息、上传音频并进行识别. 提供完整、准确的音频背景信息可以提高鸟声识别的准确率.</p>
                            <ul class="list-unstyled arrow">
                                <li> 系统目前可以识别北美四地的397种鸟类. </li>
                                <li>支持 .wav 和 .ogg 格式. </li>
                                <li>带 * 的是必填项，填写的信息越完整，识别准确率越高. </li>
                                <li>识别过程耗时较长，请耐心等候. </li>
                            </ul>
                        </div>
                    </div>
                    <!-- ============================================================== -->
                    <!-- end overview  -->
                    <!-- ============================================================== -->
                </div>

                <div class="row">
                    <!-- ============================================================== -->
                    <!-- valifation types -->
                    <!-- ============================================================== -->
                    <div class="col-xl-12 col-lg-12 col-md-12 col-sm-12 col-12">
                        <div class="card">
                            <h5 class="card-header">音频背景信息</h5>
                            <div class="card-body">
                                <form id="validationform" data-parsley-validate="" novalidate="">

                                    <div class="form-group">
                                        <label for="input-select">录音设备</label>
                                        <select class="form-control" id="input-select">
                                            <option>smartphone (iOS)</option>
											<option>smartphone (Android)</option>
                                            <option>autonomous recording units (ARU)</option>
											<option>其他设备</option>
                                        </select>
                                    </div>

                                    <div class="form-group">
                                        <label for="input-select">录音日期*</label>
                                        <div class="input-group date" id="datetimepicker4" data-target-input="nearest">
                                            <input type="text" class="form-control datetimepicker-input" data-target="#datetimepicker4" />
                                            <div class="input-group-append" data-target="#datetimepicker4" data-toggle="datetimepicker">
                                                <div class="input-group-text"><i class="far fa-calendar-alt"></i></div>
                                            </div>
                                        </div>
                                    </div>

                                    <div class="form-group">
                                        <label for="input-select">录音地点*</label>
                                        <select class="form-control" id="input-select">
                                            <option>COL - 安蒂奥基亚，哥伦比亚</option>
                                            <option>COR - 阿拉胡埃拉，哥斯达黎加</option>
                                            <option>SNE - 内华达山脉， 加利福尼亚</option>
                                            <option>SSW - 伊萨卡， 纽约</option>
                                        </select>
                                    </div>

                                    <div class="form-group">
                                        <label for="input-select">天气状况</label>
                                        <select class="form-control" id="input-select">
                                            <option>晴</option>
                                            <option>多云</option>
                                            <option>阴</option>
                                            <option>雨</option>
                                            <option>雪</option>
											<option>其他天气</option>
                                        </select>
                                    </div>

                                    <div class="form-group">
                                        <label class="">识别器精度*</label>
                                        <div class="">
                                            <input required="" type="number" min="1" max="10" placeholder="请输入 1 - 10，数字越大，识别精度越高，运算时间越长" class="form-control">
                                        </div>
                                    </div>

                                    <div class="row">
                                        <div class="col-sm-6 pb-2 pb-sm-4 pb-lg-0 pr-0">

                                        </div>
                                        <div class="col-sm-6 pl-0">
                                            <p class="text-right">
                                                <button type="submit" class="btn btn-space btn-primary">选择音频并上传</button>
                                            </p>
                                        </div>
                                    </div>

                                </form>
                            </div>
                        </div>
                    </div>
                    <!-- ============================================================== -->
                    <!-- end valifation types -->
                    <!-- ============================================================== -->
                </div>

                <div class="row">
                    <!-- ============================================================== -->
                    <!-- data table  -->
                    <!-- ============================================================== -->
                    <div class="col-xl-12 col-lg-12 col-md-12 col-sm-12 col-12">
                        <div class="card">
                            <div class="card-header">
                                <h5 class="mb-0">识别结果 - Copy, Excel, PDF, Print</h5>
                                <p>起始、终止时间格式为 “时:分:秒”， 不足5秒者，以5秒计；备选鸟种以识别置信度从高到低排列.</p>
                            </div>
                            <div class="card-body">
                                <div class="table-responsive">
                                    <table id="example" class="table table-striped table-bordered second" style="width:100%">
                                        <thead>
                                            <tr>
                                                <th>起始时间</th>
                                                <th>终止时间</th>
                                                <th>备选鸟种1</th>
                                                <th>备选鸟种2</th>
                                                <th>备选鸟种3</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr>
                                                <td>00:00:00</td>
                                                <td>00:00:00</td>
                                                <td>A</td>
                                                <td>B</td>
                                                <td>C</td>
                                            </tr>
                                        </tbody>
                                        <tfoot>
                                            <tr>
                                                <th>起始时间</th>
                                                <th>终止时间</th>
                                                <th>备选鸟种1</th>
                                                <th>备选鸟种2</th>
                                                <th>备选鸟种3</th>
                                            </tr>
                                        </tfoot>
                                    </table>
                                </div>
                            </div>
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