/*

 OpenME C/C++/Fortran/Java/PHP interface for Collective Mind Infrastructure (cM)

 OpenME - Event-driven, plugin-based interactive interface to "open up" 
          any software and connect it to cM

 Developer(s): (C) Grigori Fursin, started on 2011.09
 http://cTuning.org/lab/people/gfursin

 This library is free software; you can redistribute it and/or
 modify it under the terms of the GNU Lesser General Public
 License as published by the Free Software Foundation; either
 version 2.1 of the License, or (at your option) any later version.

 This library is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 Lesser General Public License for more details.

 You should have received a copy of the GNU Lesser General Public
 License along with this library; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

*/

package org.openme;

//Import libraries...
import java.io.*;
import java.net.*;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.JSONValue;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.net.HttpURLConnection;
import java.net.URLEncoder;

import org.apache.commons.codec.binary.Base64;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Formatter;

//Main class
public class openme
{
  // *******************************************************************
  public static JSONObject openme_store_json_file(JSONObject i, String file_name)
  {
    /*
       Store json object in file

       Input:  i         - cJSON object
               file_name - name of the file to store json object

       Output: {
                 cm_return       - return code = 0, if successful
                 (cm_error)      - error text if return code > 0
               }
    */

    JSONObject r=new JSONObject();
    BufferedWriter fp=null;

    try
    {
      fp=new BufferedWriter(new FileWriter(file_name));
    }
    catch (IOException ex)
    {
      r.put("cm_return", new Long(1));
      r.put("cm_error", "can't open json file for writing ("+ex.getMessage()+") ...");
      return r;
    }

    try
    {
       StringWriter out = new StringWriter();
       i.writeJSONString(out);
       fp.write(out.toString()); 
       fp.newLine();
    }
    catch (Exception ex)
    {
      try
      {
        fp.close();
      }
      catch (Exception ex1)
      {
         r.put("cm_return", new Long(1));
         r.put("cm_error", "can't close json file ("+ex1.getMessage()+") ...");
         return r;
      }

      r.put("cm_return", new Long(1));
      r.put("cm_error", "can't write json file ("+ex.getMessage()+") ...");
      return r;
    }

    try
    {
      fp.close();
    }
    catch (Exception ex)
    {
      r.put("cm_return", new Long(1));
      r.put("cm_error", "can't close json file ("+ex.getMessage()+") ...");
      return r;
    }

    r.put("cm_return", new Long(0));
    return r;
  }

  // *******************************************************************
  public static String[] openme_run_program(String cmd, String[] env, String path) 
  {
    /*
       FGG: TBD - call local cM

       Input:  cmd  - command line to run
               env  - list of environment variables
               path - directory where to start program

       Output: string list:
                 [0] - error text
                 [1] - command output
    */

    File dir=null;
    if (path!=null) dir=new File(path); //getCacheDir();

    Process p=null;
    String output="";
    String err="";
		
    try 
    {
      p=Runtime.getRuntime().exec(cmd,env,dir);

      BufferedReader reader=new BufferedReader(
        new InputStreamReader(p.getInputStream()));

      String line=null;
      while ((line = reader.readLine())!=null)
        output+=line+'\n';

      reader.close();
      p.waitFor();
    } 
    catch(Exception e) 
    {  
      err=e.toString();
    }

    if (p!=null) 
    {
      try
      {
        p.getOutputStream().close();
	p.getInputStream().close();
	p.getErrorStream().close(); 
      } 
      catch (IOException e) 
      {
        err=e.toString();
      }
    }
		
    return new String[] {err, output};
  }

  // *******************************************************************
  public static JSONObject openme_load_json_file(String file_name)
  {
    /*
       Load json file and create cJSON object

       Input: file_name - name of the file to load

       Output: {
                 cm_return   - return code = 0, if successful
                 (cm_error)  - error text if return code > 0
                 cm_array    - cJSON object, if success
    */

    JSONObject r=new JSONObject();
    BufferedReader fp=null;

    try
    {
      fp=new BufferedReader(new FileReader(file_name));
    }
    catch (IOException ex)
    {
      r.put("cm_return", new Long(1));
      r.put("cm_error", "can't open json file for writing ("+ex.getMessage()+") ...");
      return r;
    }

    String output="";
    try 
    {
      String line=null;
      while ((line = fp.readLine())!=null)
        output+=line+'\n';

      fp.close();
    } 
    catch (Exception ex)
    {
      try
      {
        fp.close();
      }
      catch (Exception ex1)
      {
         r.put("cm_return", new Long(1));
         r.put("cm_error", "can't close json file ("+ex1.getMessage()+") ...");
         return r;
      }

      r.put("cm_return", new Long(1));
      r.put("cm_error", "can't read json file ("+ex.getMessage()+") ...");
      return r;
    }

    try
    {
      fp.close();
    }
    catch (Exception ex)
    {
      r.put("cm_return", new Long(1));
      r.put("cm_error", "can't close json file ("+ex.getMessage()+") ...");
      return r;
    }

    JSONObject a=null;

    try
    {
      JSONParser parser=new JSONParser();
      a=(JSONObject) parser.parse(output);
    }
    catch (ParseException ex)
    {
      r.put("cm_return", new Long(1));
      r.put("cm_error", "can't parse json output ("+ex+") ...");
      return r;
    }
    
    r.put("cm_return", new Long(0));
    r.put("cm_array", a);

    return r;
  }

  // *******************************************************************
  public static JSONObject convert_str_to_sha1(JSONObject i) 
  {
    /*
    Convert string to sha1

    Input:  {
              cm_string       - string
            }

    Output: {
              cm_return             - return code >0 if not authentificated
              cm_string_sha1        - password in SHA1 (digest)
              cm_string_sha1_hex    - password in SHA1 (digest in hex)
              cm_string_sha1_base64 - BASE64 (SHA1 digest) - compatible with htpasswd format
            }
    */

    byte[] sha1 = null;
    String sha1_hex = "";
    String sha1_base64 = "";

    // Prepare return object
    JSONObject r=new JSONObject();

    String x=(String) i.get("cm_string");
    if (x==null || x=="")
    {
      r.put("cm_return", new Long(1));
      r.put("cm_error", "'cm_string' is not set in openme/convert_str_to_sha1");
      return r;
    }

    try
    {
       MessageDigest crypt = MessageDigest.getInstance("SHA-1");
       crypt.reset();
       crypt.update(x.getBytes("UTF-8"));
       sha1=crypt.digest();
       sha1_hex=byteToHexString(sha1);
       sha1_base64=new String(Base64.encodeBase64(sha1));
    }
    catch(NoSuchAlgorithmException e)
    {
      r.put("cm_return", new Long(0));
      r.put("cm_error", "can't crypt password ("+e.getMessage()+") ...");
      return r;
    }
    catch(UnsupportedEncodingException e)
    {
      r.put("cm_return", new Long(0));
      r.put("cm_error", "can't crypt password ("+e.getMessage()+") ...");
      return r;
    }

    r.put("cm_return", new Long(0));
    r.put("cm_string_sha1", sha1.toString());
    r.put("cm_string_sha1_hex", sha1_hex);
    r.put("cm_string_sha1_base64", sha1_base64);
    return r;
  }

  private static String byteToHexString(byte[] bytes)
  {
    Formatter formatter = new Formatter();

    for (byte b:bytes)
      formatter.format("%02x", b);

    return formatter.toString();
  }

  // *******************************************************************
  public static JSONObject convert_cm_array_to_uri(JSONObject i) 
  {
    /*
    Convert cM array to uri (convert various special parameters)

    Input:  {
              cm_array   - array to convert to uri
            }

    Output: {
              cm_return  - return code (check "access" function for full description)
              (cm_error) - error text  (check "access" function for full description)
              cm_string  - converted string
              cm_string1 - converted string (for urlopen)
            }
    */

    // Prepare return object
    JSONObject r=new JSONObject();

    JSONObject x=(JSONObject) i.get("cm_array");
    if (x==null)
    {
      r.put("cm_return", new Long(1));
      r.put("cm_error", "'cm_array' is not set in openme/convert_cm_array_to_uri");
      return r;
    }

    String s="";
    String s1="";

    StringWriter out = new StringWriter();
    try
    {
       x.writeJSONString(out);
    }
    catch (Exception ex)
    {
      r.put("cm_return", new Long(1));
      r.put("cm_error", "can't output json ("+ex.getMessage()+") ...");
      return r;
    }

    s=out.toString(); 

    s=s.replace("\"", "^22^");
    s=s.replace("#", "%23");

    s1=s.replace(">", "%3E");
    s1=s1.replace("<", "%3C");

    try
    {
      s=URLEncoder.encode(s, "UTF-8");
    }
    catch (Exception ex)
    {
      r.put("cm_return", new Long(1));
      r.put("cm_error", "can't encode string ("+ex.getMessage()+") ...");
      return r;
    }

    r.put("cm_return", new Long(0));
    r.put("cm_string", s);
    r.put("cm_string1", s1);

    return r;
  }

  // *******************************************************************
  public static JSONObject remote_access(JSONObject i) 
  {
    /*
    Input:  {
              cm_remote_url             - remote URL
              (cm_user_uoa)             -
              (cm_user_password)        -
              (cm_user_password1)       -
              (remote_repo_uoa)

              (remote_pre_auth)         - if 'yes', need standard http login/password pre-authentication
              (cm_remote_user_name)     - remote username
              (cm_remote_user_password) - remote password

              (cm_web_module_uoa)       - module to run
              (cm_web_action)           - action to perform
                                          if =='download', prepare entry/file download through Internet

              (cm_save_to_file)         - if cm_web_action==download,
                                          save output to this file

              (cm_console)              - if 'json', treat output as json
                                          if 'json_after_text', strip everything before json
                                          if 'txt', output to cm_stdout

              ...                       - all other request parameters

              //FGG TBD - should add support for proxy
            }

    Output: {
              cm_return   - return code = 0 if successful
                                        > 0 if error
                                        < 0 if warning (rarely used at this moment)
              (cm_error)  - error text, if cm_return > 0
              (cm_stdout) - if cm_console='txt', output there
            }
    */

    // Prepare return object
    JSONObject r=new JSONObject();

    URL u;
    HttpURLConnection c=null;  

    // Prepare request
    String x="";
    String post="";

    String con="";
    x=(String) i.get("cm_console");
    if (x!=null && x!="")
       con=x;

    String url=(String) i.get("cm_remote_url");
    if (url==null || url=="")
    {
      r.put("cm_return", new Long(1));
      r.put("cm_error", "'cm_remote_url is not defined");
      return r;
    }
    i.remove("cm_remote_url");

    x=(String) i.get("cm_user_uoa");
    if (x!=null && x!="")
    {
       if (post!="") post+="&";
       post+="cm_user_uoa="+x;
       i.remove("cm_user_uoa");
    }

    String save_to_file=(String) i.get("cm_save_to_file");
    if (save_to_file!=null)
       i.remove("cm_save_to_file");

    x=(String) i.get("cm_user_password");
    if (x!=null && x!="")
    {
       if (post!="") post+="&";
       post+="cm_user_password="+x;
       i.remove("cm_user_password");
    }

    x=(String) i.get("cm_user_password1");
    if (x!=null && x!="")
    {
       if (post!="") post+="&";
       post+="cm_user_password1="+x;
       i.remove("cm_user_password1");
    }

    x=(String) i.get("remote_repo_uoa");
    if (x!=null && x!="")
    {
       if (post!="") post+="&";
       post+="cm_repo_uoa="+x;
       i.remove("remote_repo_uoa");
    }

    // Check if needed remote pre-authentication
    String base64string="";
    x=(String) i.get("remote_pre_auth");
    if (x=="yes")
    {
       i.remove("remote_pre_auth");

       String username="";
       String password="";

       x=(String) i.get("cm_remote_user_name");
       if (x!=null && x!="")
       {
          username=x;
          i.remove("cm_remote_user_name");
       }

       x=(String) i.get("cm_remote_user_password");
       if (x!=null && x!="")
       {
          password=x;
          i.remove("cm_remote_user_password");
       }

       x=username+":"+password;
       base64string=new String(Base64.encodeBase64(x.replace("\n","").getBytes()));
    }

    // Check if data download, not json and convert it to download request
    boolean download=false;

    x=(String) i.get("cm_web_action");
    if (x=="download" || x=="show")
    {
       download=true;
       if (post!="") post+="&";
       post+="cm_web_module_uoa=web&cm_web_action="+x;
       i.remove("cm_web_action");

       if (((String) i.get("cm_web_module_uoa"))!=null) i.remove("cm_web_module_uoa");
       if (((String) i.get("cm_console"))!=null) i.remove("cm_console");
    }

    // Prepare array to transfer through Internet
    JSONObject ii=new JSONObject();
    ii.put("cm_array", i);
    JSONObject rx=convert_cm_array_to_uri(ii);
    if ((Long)rx.get("cm_return")>0) return rx;

    if (post!="") post+="&";
    post+="cm_json="+((String) rx.get("cm_string1"));

    // Prepare URL request
    String s="";
    try 
    {
      //Connect
      u=new URL(url);
      c=(HttpURLConnection) u.openConnection();

      c.setRequestMethod("POST");
      c.setRequestProperty("Content-Type", "application/x-www-form-urlencoded");
      c.setRequestProperty("Content-Length", Integer.toString(post.getBytes().length));
      if (base64string!="") c.setRequestProperty("Authorization", "Basic "+base64string);
      c.setUseCaches(false);
      c.setDoInput(true);
      c.setDoOutput(true);

      //Send request
      DataOutputStream dos=new DataOutputStream(c.getOutputStream());
      dos.writeBytes(post);
      dos.flush();
      dos.close();
    }
    catch (Exception e)
    {
      if (c!=null) c.disconnect(); 

      r.put("cm_return", new Long(1));
      r.put("cm_error", "Failed reading stream from remote server ("+e.getMessage()+") ...");
      return r;
    }

    r.put("cm_return", new Long(0));

    // Check if download, not json!
    if (download)
    {
       String name="default_download_name.dat";

       x=((String) i.get("cm_web_filename"));
       if (x!=null && x!="")
       {
         File xf = new File(x);
         name=xf.getName();
       }

       if (save_to_file!=null && save_to_file!="")  name=save_to_file;

       //Reading response in binary and at the same time saving to file
       try 
       {
         //Read response
         DataInputStream dis=new DataInputStream(c.getInputStream());
         DataOutputStream dos=new DataOutputStream(new FileOutputStream(name));

         byte[] buf=new byte[16384];

         int len;
         
         while((len=dis.read(buf))!=-1)
           dos.write(buf,0,len);

         dos.close();
         dis.close();
       }
       catch (Exception e)
       {
         if (c!=null) c.disconnect(); 

         r.put("cm_return", new Long(1));
         r.put("cm_error", "Failed reading stream from remote server or writing to file ("+e.getMessage()+") ...");
         return r;
       }
    }
    else
    {
       //Reading response in text
       try 
       {
         //Read response
         InputStream is=c.getInputStream();
         BufferedReader f=new BufferedReader(new InputStreamReader(is));
         StringBuffer ss=new StringBuffer(); 
         
         while((x=f.readLine())!=null)
         {
           ss.append(x);
           ss.append('\r');
         }

         f.close();
         s=ss.toString();
       }
       catch (Exception e)
       {
         if (c!=null) c.disconnect(); 

         r.put("cm_return", new Long(1));
         r.put("cm_error", "Failed reading stream from remote server ("+e.getMessage()+") ...");
         return r;
       }

       if (con=="json_after_text")
       {
         String json_sep="*** ### --- CM JSON SEPARATOR --- ### ***";
         int li=s.lastIndexOf(json_sep);
         if (li>=0)
         {
           s=s.substring(li+json_sep.length());
           s=s.trim();
         }
       }

       if (con=="json_after_text" || con=="json")
       {
          //Parsing json
          try
          {
            JSONParser parser=new JSONParser();
            r=(JSONObject) parser.parse(s);
          }
          catch (ParseException ex)
          {
            r.put("cm_return", new Long(1));
            r.put("cm_error", "can't parse json output ("+ex+") ...");
            return r;
          }
       }
       else
         r.put("cm_stdout", s);
    }

    if (c!=null) c.disconnect(); 

    return r;
  }

  // *******************************************************************
  public static JSONObject access(JSONObject i) 
  {
    /*
       FGG: TBD - call local cM

       Input:  i input json object

       Output: {
                 cm_return       - return code = 0, if successful
                 (cm_error)      - error text if return code > 0
                 ...
               }
    */
    File f=null;
    String fn1, fn2, fn3;

    JSONObject r=new JSONObject();

    /* Get module name */
    String rm=(String) i.get("cm_run_module_uoa");
    if (rm==null || rm=="")
    {
      r.put("cm_return", new Long(1));
      r.put("cm_error", "can't find cm_run_module_uoa in cm_action ...");
      return r;
    }
    
    /* Generate tmp files with json and for output*/
    /* First file will be deleted automatically by cM */
    try 
    {
      f=File.createTempFile("cm-", "-cm.tmp", null);
      fn1=f.getAbsolutePath();

      f=File.createTempFile("cm-", "-cm.tmp", null);
      fn2=f.getAbsolutePath();

      f=File.createTempFile("cm-", "-cm.tmp", null);
      fn3=f.getAbsolutePath();
    } 
    catch (IOException ex) 
    {
      r.put("cm_return", new Long(1));
      r.put("cm_error", "can't gerenate temp files ("+ex.getMessage()+") ...");
      return r;
    }

    /* Record input file */
    JSONObject rx=openme_store_json_file(i, fn1);
    if ((Long)rx.get("cm_return")>0) return rx;

    /* Prepare command line */
    String cmd="cmd /c cm "+rm+" @"+fn1+" > "+fn2+" 2> "+fn3;

    String[] x=openme_run_program(cmd, null, null);

    r.put("cm_stderr",x[0]);
    r.put("cm_stdout",x[1]);

    r=openme_load_json_file(fn2);
    if ((Long)r.get("cm_return")>0)
    {
      r.put("cm_return", new Long(1));
      r.put("cm_error", "output in files (STDOUT file="+fn2+"; STDERR file="+fn3+")");
      return r;
    }

    /* Remove tmp files */
    f = new File(fn2);
    f.delete();

    f = new File(fn3);
    f.delete();

    return r;
  }
}
