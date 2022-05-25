<%@ page language="java" contentType="text/html; charset=UTF-8"
         pageEncoding="UTF-8"%>
<%@ page import="java.sql.*"%>
<%@ page import="java.io.BufferedWriter" %>
<%@ page import="java.io.BufferedReader" %>
<%@ page import="java.io.FileWriter" %>
<%@ page import="java.awt.Desktop" %>
<%@ page import="java.io.File" %>


<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Insert title here</title>
</head>
<body>
<%
request. setCharacterEncoding("UTF-8");

											try {
												Class.forName("com.mysql.jdbc.Driver");
												String url = "jdbc:mysql://localhost:3306/bird?serverTimezone=UTC";
												String usernam = "root";  //���ݿ��û���
												String password = "root";  //���ݿ��û�����
												Connection conn = DriverManager.getConnection(url, usernam, password);  //����״̬

												if (conn != null) 
												{
													//usernam = request.getParameter("username");
													//password = request.getParameter("password");
													PreparedStatement pStmt = conn.prepareStatement("UPDATE flag SET isfinished = 'no' WHERE id = 'idtf';");
													pStmt.executeUpdate();
												}
												else{
													out.print("����ʧ�ܣ�");
												}
											}catch (ClassNotFoundException e) {
												e.printStackTrace();
												out.print("1");
											}catch (SQLException e){
												e.printStackTrace();
												out.print("2");
												}

    request. setCharacterEncoding("UTF-8");
	String date = request.getParameter("rdate");
	if(date.equals(""))
	{
		date = "11111111";
	}
	else
	{
		String month = date.substring(0, 2);
		String day = date.substring(3, 5);
		String year = date.substring(6, 10);
		date = year + month + day;
	}


	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	String location = request.getParameter("rlocation");
	if(location.equals("COL - Antioquia, Colombia"))
	{
		location = "COL";
	}
	else if(location.equals("COR - Alajuela, Costa Rica"))
	{
		location = "COR";
	}
	else if(location.equals("SNE - Sierra Nevada, California"))
	{
		location = "SNE";
	}
	else if(location.equals("SSW - Ithaca, New York"))
	{
		location = "SSW";
	}
	else
	{
		location = "COL";
	}




	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	String number = request.getParameter("mn");
	if(number.equals(""))
	{
		number = "1";
		out.print("<script language='javaScript'> alert('You must feed a value for \"identification precision*\"');</script>");
		out.print("<script language='javaScript'> history.go(-1);</script>");
		//response.setHeader("refresh", "0;url=index.jsp");
	}


	/*	out.print(date);
		out.print(" - ");
		out.print(location);
		out.print(" - ");
		out.print(number);


		BufferedWriter file = new BufferedWriter(new FileWriter("../webapps/ROOT/record.txt"));
		file.write(location + "@" + date  + "@" + number);
		file.close();*/

															try {
												Class.forName("com.mysql.jdbc.Driver");
												String url = "jdbc:mysql://localhost:3306/bird?serverTimezone=UTC";
												String usernam = "root";  //���ݿ��û���
												String password = "root";  //���ݿ��û�����
												Connection conn = DriverManager.getConnection(url, usernam, password);  //����״̬

												if (conn != null) 
												{
													//usernam = request.getParameter("username");
													//password = request.getParameter("password");
													PreparedStatement pStmt = conn.prepareStatement("delete from uinput;");
													pStmt.executeUpdate();
												}
												else{
													out.print("����ʧ�ܣ�");
												}
											}catch (ClassNotFoundException e) {
												e.printStackTrace();
												out.print("1");
											}catch (SQLException e){
												e.printStackTrace();
												out.print("2");
												}



													try {
												Class.forName("com.mysql.jdbc.Driver");
												String url = "jdbc:mysql://localhost:3306/bird?serverTimezone=UTC";
												String usernam = "root";  //���ݿ��û���
												String password = "root";  //���ݿ��û�����
												Connection conn = DriverManager.getConnection(url, usernam, password);  //����״̬

												if (conn != null) 
												{
													//usernam = request.getParameter("username");
													//password = request.getParameter("password");
													PreparedStatement pStmt = conn.prepareStatement("INSERT INTO uinput ( rlocation, rdate, rnumber ) VALUES ('" + location + "', '" + date + "','" + number + "' );");
													pStmt.executeUpdate();
												}
												else{
													out.print("����ʧ�ܣ�");
												}
											}catch (ClassNotFoundException e) {
												e.printStackTrace();
												out.print("1");
											}catch (SQLException e){
												e.printStackTrace();
												out.print("2");
												}

		//Thread.sleep(10000);
		out.print("Identifying... Please Wait...");

		/*����exe*/
		String flag = "test";
		BufferedReader bufferedReader = null;
        try {
            Desktop.getDesktop().open(new File("D:/birdclefFirstPlace/dist/osop.exe"));
        } catch (Exception ex) {
            ex.printStackTrace();
			out.print("no exe");
			flag = "no";
        } finally {
            if (bufferedReader != null) {
                try {
                    bufferedReader.close();
                } catch (Exception ex) {
                }
            }
        }

		//response.sendRedirect("index2.jsp");
		//if (flag.equals("test"))
			//response.sendRedirect("index2.jsp");


		String flg = "no";
		while(true)
		{
			Thread.sleep(10000);
			try 
			{
												Class.forName("com.mysql.jdbc.Driver");
												String url = "jdbc:mysql://localhost:3306/bird?serverTimezone=UTC";
												String usernam = "root";  //���ݿ��û���
												String password = "root";  //���ݿ��û�����
												Connection conn = DriverManager.getConnection(url, usernam, password);  //����״̬

												if (conn != null) 
												{
													//usernam = request.getParameter("username");
													//password = request.getParameter("password");
													PreparedStatement pStmt = conn.prepareStatement("select * from flag where id='idtf';");
													ResultSet rs = pStmt.executeQuery();
													int i = 0;
													while(rs.next())
													{
														String idd = rs.getObject("id").toString();
														String isfin = rs.getObject("isfinished").toString();
														if (isfin.equals("yes"))
														{
															flg = "yes";
														}
								
													}

												}
												else
												{
													out.print("����ʧ�ܣ�");
												}
			}
			catch (ClassNotFoundException e) 
			{
				e.printStackTrace();
				out.print("11");
			}
			catch (SQLException e)
			{
				e.printStackTrace();
				out.print("22");
			}

			if(flg.equals("yes"))
			{
				response.sendRedirect("index2.jsp");	
				break;
			}
		}
		


 %>
</body>
</html>