<%@ page language="java" contentType="text/html; charset=UTF-8"
         pageEncoding="UTF-8"%>
<%@ page import="java.sql.*"%>
<%@ page import="java.io.BufferedWriter" %>
<%@ page import="java.io.FileWriter" %>
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Insert title here</title>
</head>
<body>
<%
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
	if(location.equals("COL - ���ٰ»��ǣ����ױ���"))
	{
		location = "COL";
	}
	else if(location.equals("COR - ��������������˹�����"))
	{
		location = "COR";
	}
	else if(location.equals("SNE - �ڻ���ɽ���� ����������"))
	{
		location = "SNE";
	}
	else if(location.equals("SSW - �������� ŦԼ"))
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


	out.print(date);
	out.print(" - ");
	out.print(location);
	out.print(" - ");
	out.print(number);


	BufferedWriter file = new BufferedWriter(new FileWriter("../webapps/ROOT/record.txt"));
	file.write(location + "@" + date  + "@" + number);
	file.close();





 %>
</body>
</html>