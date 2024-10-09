FROM openjdk:21

RUN mkdir -p deploy
WORKDIR /deploy

COPY ./build/libs/LogServer-0.0.1-SNAPSHOT.jar LogServer.jar

ENTRYPOINT ["java","-jar","/deploy/LogServer.jar"]