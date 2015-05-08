(defproject logistic "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.5.1"]

                 [ring "1.3.2"]
                 [compojure "1.3.3"]
                 [javax.servlet/servlet-api "2.5"] ;recommended in ring github

                 [org.clojure/math.numeric-tower "0.0.4"]
                 [org.clojure/data.csv "0.1.2"]
                 [slingshot "0.12.2"]
                 [org.jblas/jblas "1.2.3"]
                 [net.mikera/core.matrix "0.34.0"]
                 [clatrix "0.4.0"]

                 [org.clojure/tools.trace "0.7.8"]

                 [org.clojure/tools.logging "0.3.1"]

                 [log4j "1.2.17" :exclusions [javax.mail/mail
                                              javax.jms/jms
                                              com.sun.jdmk/jmxtools
                                              com.sun.jmx/jmxri]]]
  
  :target-path "target/%s"

  :plugins [[lein-ring "0.8.10"]
  		    [lein-exec "0.3.2"]
  		    [lein-ancient "0.5.4"] ]

  :ring {:handler logistic.core/app-handler}

  :main logistic.core

  :jvm-opts ["-server"
             "-Xms4G"
             "-Xmx4G"
             "-XX:NewRatio=5"
             "-XX:+UseConcMarkSweepGC"
             "-XX:+UseParNewGC"
             "-XX:MaxPermSize=64m"])


