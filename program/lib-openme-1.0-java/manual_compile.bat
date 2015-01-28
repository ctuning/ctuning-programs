rem set CLASSPATH=.;json-simple-1.1.1.jar;%CLASSPATH%

call cm_code_env_347be8de98cddabe.bat

javac org/openme/openme.java
jar cf openme.jar org\openme\openme.class
