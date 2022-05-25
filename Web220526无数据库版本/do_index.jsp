<%@ page language="java" contentType="text/html; charset=UTF-8"
         pageEncoding="UTF-8"%>
<%@ page import="java.sql.*"%>
<%@ page import="java.io.BufferedWriter" %>
<%@ page import="java.io.BufferedReader" %>
<%@ page import="java.io.FileWriter" %>
<%@ page import="java.io.FileReader" %>
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
    String web_path = "D:/tomcat/webapps/ROOT/";
    String flg2 = "no";
    BufferedReader buf2 = null ;		// 声明对象
    buf2 = new BufferedReader(new FileReader(new File(web_path+"isuploaded.txt"))) ;	// 将字节流变为字符流
    try{
        flg2 = buf2.readLine() ;	// 读取一行数据
    }catch(Exception e){
        e.printStackTrace() ;	// 输出信息
    }

/////////////////////
if(flg2.equals("yes")){
//////////////////////
    BufferedWriter flagfile = new BufferedWriter(new FileWriter(web_path+"isfinished.txt"));
	flagfile.write("no");
	flagfile.close();

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
		//out.print("<script language='javaScript'> alert('You must feed a value for \"identification precision*\"');</script>");
		//out.print("<script language='javaScript'> history.go(-1);</script>");
		//response.setHeader("refresh", "0;url=index.jsp");
	}


		out.print(date);
		out.print(" - ");
		out.print(location);
		out.print(" - ");
		out.print(number);


		BufferedWriter file = new BufferedWriter(new FileWriter(web_path+"record.txt"));
		file.write(location + "@" + date  + "@" + number);
		file.close();


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


		String flg = "no";
		while(true)
		{

            Thread.sleep(10000);
            BufferedReader buf = null ;		// 声明对象
            buf = new BufferedReader(new FileReader(new File(web_path+"isfinished.txt"))) ;	// 将字节流变为字符流
            try{
                flg = buf.readLine() ;	// 读取一行数据
            }catch(Exception e){
                e.printStackTrace() ;	// 输出信息
            }


			if(flg.equals("yes"))
			{
				response.sendRedirect("index2.jsp");
				break;
			}
		}
		

//////////////////
}
else
{
    out.print("<script language='javaScript'> alert('Please upload a file!');</script>");
	out.print("<script language='javaScript'> history.go(-1);</script>");
}
//////////////////

 %>
</body>
</html>